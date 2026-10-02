"""Exercise two independent text lanes, queueing and disconnect recovery.

Requires an idle CUDA device and an explicit compatible GGUF. Starts/stops only
its own loopback server. This is a state-isolation regression, not a benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--device', default='CUDA0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command = [args.server, '-m', args.model, '--host', '127.0.0.1', '--port', str(port),
               '-c', '8192', '--device', args.device, '--parallel', '2', '--threads-http', '8',
               '--kvmem', '--kvmem-budget', '2048', '--kvmem-gen-reserve', '1024',
               '--kvmem-cpu-gb', '2', '--kvmem-nvme-gb', '0',
               '-fa', 'on', '--reasoning-effort', 'none', '--temp', '0',
               '--presence-penalty', '0', '--no-webui',
               '--spec-type', 'draft-mtp' if args.mtp else 'none']
    evidence = {'command': command, 'requests': [], 'checks': []}

    def check(name, passed):
        evidence['checks'].append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    def get(path):
        with opener.open(base + path, timeout=5) as response:
            return json.load(response)

    def generate(payload):
        request = urllib.request.Request(base + '/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=180) as response:
            record = {'lane': response.headers.get('X-KVMem-Lane'),
                      'response': json.load(response)}
        return record

    def payload(secret, limit=64):
        return {'messages': [
            {'role': 'system', 'content': f'The secret word for this conversation is {secret}.'},
            {'role': 'user', 'content': ('Unrelated background text. ' * 850) +
             ' What is the secret word? Reply with the word only.'}],
            'temperature': 0, 'seed': 7, 'max_tokens': limit,
            'reasoning_effort': 'none', 'cache_reset': True}

    def content(record):
        return record['response']['choices'][0]['message']['content'].strip()

    with (args.output / 'server.log').open('wb') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 360
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server exited during startup; see server.log')
                try:
                    if get('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('server startup')
                time.sleep(.5)
            check('two slots exposed', len(get('/slots')) == 2)
            check('requested speculative backend',
                  all(s['speculative'] == args.mtp for s in get('/slots')))
            secrets = ['AMBER', 'VIOLET']
            baseline = [generate(payload(secret)) for secret in secrets]
            evidence['requests'].extend(baseline)
            for secret, record in zip(secrets, baseline):
                check('isolated recall ' + secret, content(record) == secret)
            saw_both_busy = False
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(generate, payload(secret)) for secret in secrets]
                while not all(f.done() for f in futures):
                    saw_both_busy |= all(s['is_processing'] for s in get('/slots'))
                    time.sleep(.025)
                concurrent = [f.result() for f in futures]
                evidence['requests'].extend(concurrent)
                check('two active contexts observed', saw_both_busy)
                check('different physical lanes', {r['lane'] for r in concurrent} == {'0', '1'})
                check('concurrent output matches isolated',
                      list(map(content, concurrent)) == list(map(content, baseline)))
                # Reuse both lanes and force a third caller to wait for admission.
                more = list(pool.map(generate, [payload('AMBER'), payload('VIOLET'), payload('AMBER')]))
                evidence['requests'].extend(more)
                check('queued requests preserve state', list(map(content, more)) == ['AMBER', 'VIOLET', 'AMBER'])

                # Short recall alone can overlap only prefill. Keep both lanes
                # generating long enough to exercise independent decode/MTP state.
                long_prompts = []
                for secret in secrets:
                    item = payload(secret, 512)
                    item['messages'][-1]['content'] = item['messages'][-1]['content'].replace(
                        ' What is the secret word? Reply with the word only.', '') + (
                        ' First print the secret word, then count every integer from 1 '
                        'to 10000. Do not skip or abbreviate.')
                    long_prompts.append(item)
                serial_long = [generate(item) for item in long_prompts]
                paired_long = list(pool.map(generate, long_prompts))
                evidence['requests'].extend(serial_long + paired_long)
                check('sustained decode workload', all(
                    r['response']['usage']['completion_tokens'] == 512
                    for r in serial_long + paired_long))
                check('sustained decode uses both lanes',
                      {r['lane'] for r in paired_long} == {'0', '1'})
                check('sustained concurrent output matches isolated',
                      list(map(content, paired_long)) == list(map(content, serial_long)))
                check('sustained decode preserves secret', all(
                    content(r).startswith(secret) for secret, r in zip(secrets, paired_long)))

            stream_payload = {'messages': [{'role': 'user', 'content':
                'Count every integer from 1 to 10000. Do not skip or abbreviate.'}],
                'max_tokens': 4096, 'stream': True, 'temperature': 0,
                'reasoning_effort': 'none', 'cache_reset': True}
            request = urllib.request.Request(base + '/v1/chat/completions',
                data=json.dumps(stream_payload).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(request, timeout=180) as response:
                saw_token = False
                for line in response:
                    if not line.startswith(b'data:') or line.strip() == b'data: [DONE]':
                        continue
                    event = json.loads(line[5:])
                    if any(choice.get('delta', {}).get('content')
                           for choice in event.get('choices', [])):
                        saw_token = True
                        break
                check('disconnect occurs during decode', saw_token)
            deadline = time.monotonic() + 30
            while any(s['is_processing'] for s in get('/slots')):
                if time.monotonic() >= deadline:
                    raise TimeoutError('disconnected request retained its lane')
                time.sleep(.1)
            recovered = generate(payload('VIOLET'))
            evidence['requests'].append(recovered)
            check('inference after disconnect', content(recovered) == 'VIOLET')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            evidence['exit_code'] = proc.returncode
            (args.output / 'results.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(evidence['checks'])}))


if __name__ == '__main__':
    main()
