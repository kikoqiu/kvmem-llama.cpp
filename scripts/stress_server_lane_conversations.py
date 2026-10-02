"""Local real-model load test for dynamic lanes and conversation isolation.

No downloads. Starts and terminates only its own loopback server. Request
latencies include admission/queue time; token throughput excludes canceled work.
Use --cuda-visible with PCI ordering to select the same GPU as nvidia-smi.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


WORDS = ('AMBER VIOLET GREEN BLUE RED YELLOW ORANGE PURPLE SILVER GOLD WHITE '
         'BLACK BROWN PINK CYAN MAGENTA TEAL INDIGO CORAL OLIVE PEACH LIME '
         'NAVY RUBY').split()


def percentiles(values):
    if not values:
        return {}
    values = sorted(values)
    return {f'p{p}': round(values[min(len(values) - 1, int((len(values) - 1) * p / 100))], 3)
            for p in (50, 95, 99)}


def process_memory(pid):
    if os.name != 'nt':
        return {}

    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_working', 'working', 'peak_paged',
            'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x410, 0, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        data = Counters()
        data.cb = ctypes.sizeof(data)
        if not kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(data), data.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'working_mib': round(data.working / 2**20, 2),
                'private_mib': round(data.private / 2**20, 2)}
    finally:
        kernel.CloseHandle(handle)


def gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, name, memory, utilization, temperature = (x.strip() for x in line.split(','))
        rows.append({'index': int(index), 'name': name, 'used_mib': int(memory),
                     'utilization': int(utilization), 'temperature': int(temperature)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clients', default='2,4,8,16')
    parser.add_argument('--parallel', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--ramp-seconds', type=float, default=60)
    parser.add_argument('--soak-seconds', type=float, default=600)
    parser.add_argument('--conversations', type=int, default=8)
    parser.add_argument('--logical-conversations', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=192)
    parser.add_argument('--cuda-visible', default='0')
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--cpu-gb', default='0.0625')
    parser.add_argument('--request-timeout', type=float, default=120)
    parser.add_argument('--sanitizer', type=Path, help='optional compute-sanitizer executable for diagnosis')
    parser.add_argument('--sanitizer-kernel', help='optional mangled kernel name regex to check')
    args = parser.parse_args()
    clients = [int(x) for x in args.clients.split(',')]
    if not clients or min(clients) < 1 or not args.parallel <= args.conversations <= args.logical_conversations <= len(WORDS):
        parser.error('positive clients and parallel <= host stores <= logical conversations <= 24 required')
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(args.server.resolve()), '-m', str(args.model.resolve()), '--device', 'CUDA0',
        '--host', '127.0.0.1', '--port', str(port), '--parallel', str(args.parallel), '--threads-http', str(max(clients) + 16),
        '-c', '8192', '--kvmem-conversations', str(args.conversations), '--kvmem-cpu-gb', args.cpu_gb,
        '--kvmem-budget', '2048', '--kvmem-gen-reserve', '1024', '--kvmem-nvme-gb', '0',
        '--kvmem-query-policy', 'user', '-fa', 'on', '--reasoning-effort', 'none', '--temp', '0',
        '--presence-penalty', '0', '--spec-type', 'none', '--no-webui', '-t', '4', '-tb', '4']
    if args.sanitizer:
        prefix = [str(args.sanitizer.resolve()), '--tool', 'memcheck', '--error-exitcode', '99',
                  '--log-file', str((args.output / 'sanitizer.log').resolve())]
        if args.sanitizer_kernel:
            prefix += ['--kernel-name', 'regex=' + args.sanitizer_kernel]
        command = prefix + command
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.cuda_visible, KVMEM_TRACE='0')
    started = time.monotonic()
    lock = threading.Lock()
    stop = threading.Event()
    state = {'phase': 'startup'}
    records, samples = [], []
    summary = {'command': command, 'environment': {k: env.get(k) for k in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES',
               'GGML_CUDA_DISABLE_GRAPHS', 'LLAMA_GRAPH_REUSE_DISABLE', 'KVMEM_DECODE_GRAPH_SLOTS',
               'GGML_CUDA_PDL', 'CUDA_LAUNCH_BLOCKING')},
               'args': vars(args).copy(), 'phases': [], 'passed': False}
    summary['args'] = {k: str(v) if isinstance(v, Path) else v for k, v in summary['args'].items()}
    baseline_hashes = {}
    opener_local = threading.local()

    def opener():
        if not hasattr(opener_local, 'value'):
            opener_local.value = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener_local.value

    def get(path, timeout=10):
        with opener().open(base + path, timeout=timeout) as response:
            return json.load(response)

    def idle(timeout=60):
        deadline = time.monotonic() + timeout
        while any(slot['is_processing'] for slot in get('/slots')):
            if time.monotonic() >= deadline:
                raise TimeoutError('lanes did not recover after drain/disconnect')
            time.sleep(.1)

    def messages(index, kind):
        word = WORDS[index]
        background = '\n'.join(f'Conversation {index:02d} routing note {i:03d}: this entry is stable.' for i in range(110))
        instruction = ('What is the secret word? Reply with the word only.' if kind == 'short' else
            'Start your answer with the secret word in uppercase, then count every integer from 1 to 10000, '
            'separated by commas. Do not skip, abbreviate, or stop early.')
        return [{'role': 'user', 'content': background + f'\nThe secret word is {word}.\n' + instruction}]

    def save(record):
        with lock:
            records.append(record)
            request_log.write(json.dumps(record) + '\n')
            request_log.flush()
            if record.get('error'):
                stop.set()

    def generate(index, kind, phase, cancel=False):
        begin = time.monotonic()
        record = {'phase': phase, 'conversation': index, 'kind': kind, 'cancel_requested': cancel,
                  'start_s': round(begin - started, 3), 'outcome': 'error'}
        content, timings, saw_done = '', {}, False
        payload = {'messages': messages(index, kind), 'kvmem': {'conversation_id': f'STRESS_{index:02d}'},
            'temperature': 0, 'seed': 7, 'reasoning_effort': 'none', 'stream': True,
            'max_tokens': 512 if cancel else (16 if kind == 'short' else args.max_tokens)}
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with opener().open(request, timeout=args.request_timeout) as response:
                record['status'] = response.status
                record['lane'] = response.headers.get('X-KVMem-Lane')
                if record['lane'] not in {str(index) for index in range(args.parallel)}:
                    raise AssertionError('invalid physical lane')
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    if line.strip() == b'data: [DONE]':
                        saw_done = True
                        break
                    event = json.loads(line[5:])
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    timings = event.get('timings', timings)
                    for choice in event.get('choices', []):
                        piece = choice.get('delta', {}).get('content')
                        if piece:
                            record.setdefault('ttft_s', round(time.monotonic() - begin, 6))
                            content += piece
                    if cancel and timings.get('predicted_n', 0) >= 24:
                        record['outcome'] = 'canceled'
                        break
            word = WORDS[index]
            head = content[:80].upper()
            other_words = [w for w in WORDS if w != word and re.search(r'\b' + w + r'\b', head)]
            if not re.search(r'\b' + word + r'\b', head) or other_words:
                raise AssertionError(f'conversation isolation: expected {word}, got {content[:160]!r}')
            if record['outcome'] != 'canceled':
                if not saw_done:
                    raise AssertionError('stream ended without DONE')
                if kind == 'long' and timings.get('predicted_n', 0) < 64:
                    raise AssertionError('long generation too short to stress decode')
                record['outcome'] = 'completed'
            record['tokens'] = timings.get('predicted_n', 0)
            record['cache_tokens'] = timings.get('cache_n', 0)
            record['prompt_tokens'] = timings.get('prompt_n', 0) + timings.get('cache_n', 0)
            record['content_sha256'] = hashlib.sha256(content.encode()).hexdigest()
            record['content_prefix'] = content[:200]
            reference = baseline_hashes.get((index, kind))
            if reference and not cancel:
                record['matches_serial_baseline'] = reference == record['content_sha256']
                if not record['matches_serial_baseline']:
                    raise AssertionError('output differs from serial baseline')
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            record['content_prefix'] = content[:200]
            if isinstance(error, urllib.error.HTTPError):
                record['status'] = error.code
        record['latency_s'] = round(time.monotonic() - begin, 6)
        save(record)
        return record

    def reject(phase, oversized=False):
        payload = {'messages': [{'role': 'user', 'content': 'x ' * 12000 if oversized else 'hello'}],
                   'max_tokens': 16, 'kvmem': {'conversation_id': 'INVALID_STRESS'}}
        if not oversized:
            payload['cache_reset'] = 'invalid'
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        begin = time.monotonic()
        record = {'phase': phase, 'kind': 'oversized' if oversized else 'invalid', 'outcome': 'error'}
        try:
            with opener().open(request, timeout=args.request_timeout) as response:
                record['status'] = response.status
            record['error'] = 'invalid request unexpectedly accepted'
        except urllib.error.HTTPError as error:
            record['status'] = error.code
            if error.code == 400:
                record['outcome'] = 'expected_rejection'
            else:
                record['error'] = f'expected HTTP 400, got {error.code}'
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        record['latency_s'] = round(time.monotonic() - begin, 6)
        save(record)

    def observe():
        next_gpu = 0
        gpu = []
        while not stop.wait(1):
            try:
                slots = get('/slots', timeout=5)
                stores = slots[0]['kvmem']['conversations']
                if len(slots) != args.parallel or stores['count'] > args.conversations:
                    raise AssertionError('physical lane/global host-store cap violated')
                if time.monotonic() >= next_gpu:
                    gpu = gpu_memory()
                    next_gpu = time.monotonic() + 5
                sample = {'elapsed_s': round(time.monotonic() - started, 3), 'phase': state['phase'],
                    'active_lanes': sum(slot['is_processing'] for slot in slots),
                    'stores': stores, 'gpu': gpu, **process_memory(proc.pid)}
                with lock:
                    samples.append(sample)
                    sample_log.write(json.dumps(sample) + '\n')
                    sample_log.flush()
            except Exception as error:
                save({'phase': state['phase'], 'kind': 'monitor', 'outcome': 'error', 'error': str(error)})

    def run_phase(name, count, duration, churn):
        state['phase'] = name
        begin = time.monotonic()
        deadline = begin + duration

        def worker(number):
            rng = random.Random(1701 + count * 31 + number)
            while time.monotonic() < deadline and not stop.is_set():
                total = args.logical_conversations if churn else args.conversations
                index = rng.randrange(min(4, total)) if rng.random() < .25 else rng.randrange(total)
                kind = 'short' if rng.random() < .2 else 'long'
                generate(index, kind, name, cancel=kind == 'long' and rng.random() < .06)

        print(json.dumps({'event': 'phase_start', 'phase': name, 'clients': count, 'seconds': duration}), flush=True)
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(worker, i) for i in range(count)]
            next_invalid = begin + 15
            next_report = begin + 30
            invalid_count = 0
            while time.monotonic() < deadline and not stop.is_set():
                now = time.monotonic()
                if now >= next_invalid:
                    reject(name, oversized=invalid_count % 2 == 1)
                    invalid_count += 1
                    next_invalid = now + 15
                if now >= next_report:
                    with lock:
                        current = [r for r in records if r['phase'] == name]
                    print(json.dumps({'event': 'progress', 'phase': name,
                        'elapsed_s': round(now - begin), 'outcomes': dict(Counter(r['outcome'] for r in current))}), flush=True)
                    next_report = now + 30
                stop.wait(.2)
            for future in futures:
                future.result()
        idle()
        elapsed = time.monotonic() - begin
        with lock:
            current = [r for r in records if r['phase'] == name]
            readings = [s for s in samples if s['phase'] == name]
        complete = [r for r in current if r['outcome'] == 'completed']
        generated = sum(r['tokens'] for r in complete)
        latencies = [r['latency_s'] for r in complete]
        first_tokens = [r['ttft_s'] for r in complete if 'ttft_s' in r]
        phase = {'name': name, 'clients': count, 'duration_s': round(elapsed, 3),
            'outcomes': dict(Counter(r['outcome'] for r in current)),
            'tokens': generated, 'tokens_per_s': round(generated / elapsed, 2),
            'requests_per_s': round(len(complete) / elapsed, 3),
            'ttft_s': percentiles(first_tokens), 'latency_s': percentiles(latencies),
            'lanes': dict(Counter(r['lane'] for r in complete)),
            'cache_hit_requests': sum(r['cache_tokens'] > 0 for r in complete),
            'baseline_matches': sum(r.get('matches_serial_baseline', False) for r in complete),
            'all_busy_samples': sum(s['active_lanes'] == args.parallel for s in readings),
            'monitor_samples': len(readings), 'end_slots': get('/slots')}
        for key in ('working_mib', 'private_mib'):
            values = [s[key] for s in readings if key in s]
            if values:
                phase[key] = {'begin_median': statistics.median(values[:10]),
                              'end_median': statistics.median(values[-10:]), 'peak': max(values)}
        target = [g for s in readings for g in s['gpu'] if g['index'] == args.gpu_index]
        if target:
            phase['gpu'] = {'used_mib_peak': max(g['used_mib'] for g in target),
                            'temperature_peak': max(g['temperature'] for g in target),
                            'utilization_mean': round(statistics.mean(g['utilization'] for g in target), 1)}
        summary['phases'].append(phase)
        print(json.dumps({'event': 'phase_done', **{k: v for k, v in phase.items() if k != 'end_slots'}}), flush=True)
        if stop.is_set():
            raise AssertionError('load test failed; inspect requests.jsonl')

    with (args.output / 'server.log').open('wb') as log, \
         (args.output / 'requests.jsonl').open('w', encoding='utf-8') as request_log, \
         (args.output / 'samples.jsonl').open('w', encoding='utf-8') as sample_log:
        summary['gpu_before'] = gpu_memory()
        summary['target_gpu'] = next(row for row in summary['gpu_before'] if row['index'] == args.gpu_index)
        proc = subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        monitor = None
        summary['pid'] = proc.pid
        try:
            deadline = time.monotonic() + 360
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    if get('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup')
                time.sleep(.5)
            summary['props'] = get('/props')
            assert summary['props']['total_slots'] == args.parallel
            assert summary['props']['kvmem']['conversations'] == args.conversations
            monitor = threading.Thread(target=observe, daemon=True)
            monitor.start()
            state['phase'] = 'serial_calibration'
            for index in range(args.logical_conversations):
                for kind in ('short', 'long'):
                    result = generate(index, kind, state['phase'])
                    if result.get('error'):
                        raise AssertionError(result['error'])
                    baseline_hashes[index, kind] = result['content_sha256']
            print(json.dumps({'event': 'calibration_passed', 'requests': len(baseline_hashes)}), flush=True)
            for count in clients:
                run_phase(f'ramp_{count}', count, args.ramp_seconds, churn=False)
            run_phase('churn_soak', max(clients), args.soak_seconds, churn=True)
            state['phase'] = 'recovery'
            for index in range(args.logical_conversations):
                if generate(index, 'short', state['phase']).get('error'):
                    raise AssertionError('post-load recovery failed')
            idle()
            before = get('/slots')[0]['kvmem']['conversations']
            reject(state['phase'])
            reject(state['phase'], oversized=True)
            after = get('/slots')[0]['kvmem']['conversations']
            assert (before['switches'], before['evictions']) == (after['switches'], after['evictions'])
            summary['final_slots'] = get('/slots')
            if max(clients) >= args.parallel:
                used = {lane for phase in summary['phases'] for lane in phase['lanes']}
                summary['all_lanes_exercised'] = used == {str(index) for index in range(args.parallel)}
                assert summary['all_lanes_exercised'], 'some physical lanes served no completed load requests'
                assert any(phase['all_busy_samples'] for phase in summary['phases']), 'no sample had all lanes active'
            summary['memory_after_load'] = process_memory(proc.pid)
            # Let outstanding socket cleanup complete before the final memory reading.
            for _ in range(10):
                if stop.wait(1):
                    raise AssertionError('monitor reported an error')
            summary['memory_after_idle'] = process_memory(proc.pid)
            summary['passed'] = not any(r.get('error') for r in records)
        except Exception as error:
            summary['failure'] = f'{type(error).__name__}: {error}'
            print(json.dumps({'event': 'failure', 'error': summary['failure']}), flush=True)
        finally:
            stop.set()
            if monitor:
                monitor.join(timeout=20)
            summary['server_alive_before_cleanup'] = proc.poll() is None
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            summary['server_exit_code_after_cleanup'] = proc.returncode
            summary['duration_s'] = round(time.monotonic() - started, 3)
            summary['outcomes'] = dict(Counter(r['outcome'] for r in records))
            summary['errors'] = [r for r in records if r.get('error')]
            summary['gpu_after_cleanup'] = gpu_memory()
            summary['monitor_samples'] = len(samples)
            summary['memory_peak'] = {key: max((s[key] for s in samples if key in s), default=0)
                                      for key in ('working_mib', 'private_mib')}
            (args.output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', 'passed': summary['passed'], 'duration_s': summary['duration_s'],
                      'outcomes': summary['outcomes'], 'errors': len(summary['errors'])}), flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
