"""Real-model regression for dynamic lanes and a global conversation pool.

No downloads. Owns only its ephemeral loopback server. Optional synthetic
images exercise a shared CPU/GPU projector; --mtp exercises follower state.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib


def image_url(rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 512, 512, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + bytes(rgb) * 512) * 512)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='CUDA0')
    p.add_argument('--parallel', type=int, choices=(2, 3, 4), default=2)
    p.add_argument('--cpu-gb', default='0.125')
    p.add_argument('--soft-gb', type=float, default=0)
    p.add_argument('--conversations', type=int, default=3)
    p.add_argument('--mmproj', type=Path)
    p.add_argument('--mmproj-device', default='none')
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--query-policy', choices=['user', 'legacy'], default='user')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(a.server.resolve()), '-m', str(a.model.resolve()), '--device', a.device,
        '--host', '127.0.0.1', '--port', str(port), '-c', '8192', '--parallel', str(a.parallel),
        '--threads-http', '8', '--kvmem-query-policy', a.query_policy, '--kvmem-conversations', str(a.conversations),
        '--kvmem-cpu-gb', a.cpu_gb, '--kvmem-conversations-gb', str(a.soft_gb), '--kvmem-budget', '2048', '--kvmem-gen-reserve', '1024',
        '-fa', 'on', '--reasoning-effort', 'none', '--temp', '0', '--presence-penalty', '0',
        '--spec-type', 'draft-mtp' if a.mtp else 'none', '--no-webui', '-t', '4', '-tb', '4']
    if a.mmproj:
        command += ['--mmproj', str(a.mmproj.resolve()), '--mmproj-device', a.mmproj_device,
                    '--image-min-tokens', '256', '--image-max-tokens', '512']
    env = dict(os.environ, KVMEM_TRACE='1')
    evidence = {'command': command, 'checks': [], 'requests': []}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check(name, ok):
        evidence['checks'].append({'name': name, 'pass': bool(ok)})
        print(('PASS ' if ok else 'FAIL ') + name, flush=True)
        if not ok:
            raise AssertionError(name)

    def request(path, payload=None):
        req = urllib.request.Request(base + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=300) as r:
                return r.status, json.load(r), r.headers.get('X-KVMem-Lane')
        except urllib.error.HTTPError as e:
            return e.code, json.load(e), None

    def get(path):
        return request(path)[1]

    def stats():
        return get('/slots')[0]['kvmem']['conversations']

    def ask(name, messages, reset=False):
        status, response, lane = request('/v1/chat/completions', {
            'messages': messages, 'kvmem': {'conversation_id': name},
            'max_tokens': 32, 'temperature': 0, 'reasoning_effort': 'none', 'cache_reset': reset})
        record = {'name': name, 'lane': lane, 'response': response}
        evidence['requests'].append(record)
        check('request succeeds ' + name, status == 200)
        check('hard global store cap', stats()['count'] <= max(a.parallel, a.conversations))
        return record

    def text(record):
        return record['response']['choices'][0]['message']['content'].strip()

    def hit(record):
        return record['response']['usage']['prompt_cache_hit_tokens']

    codes = dict(zip('ABCDE'[:a.parallel + 1], ('AMBER', 'VIOLET', 'GREEN', 'BLUE', 'RED')))
    colors = {'A': (255, 0, 0), 'B': (0, 0, 255), 'C': (0, 255, 0),
              'D': (255, 255, 0), 'E': (255, 255, 255)}
    histories = {}
    for name, code in codes.items():
        background = '\n'.join(f'Channel {name} routing note {i:03d}: this entry is stable.' for i in range(110))
        content = background + f'\nThe secret word is {code}. What is the secret word? Reply with the word only.'
        if a.mmproj:
            content = [{'type': 'image_url', 'image_url': {'url': image_url(colors[name])}},
                       {'type': 'text', 'text': background + '\nWhat is the dominant color of the picture? One word only.'}]
        histories[name] = [{'role': 'user', 'content': content}]

    def append_followup(name, record):
        histories[name] += [{'role': 'assistant', 'content': text(record)}, {'role': 'user', 'content':
            'Repeat the picture color. One word only.' if a.mmproj else 'Repeat the secret word. One word only.'}]

    def long_stream(name):
        payload = {'messages': [{'role': 'user', 'content':
            'Count every integer from 1 to 10000, separated by commas. Do not abbreviate or stop early.'}],
            'max_tokens': 768, 'stream': True, 'temperature': 0, 'reasoning_effort': 'none',
            'cache_reset': True, 'kvmem': {'conversation_id': name}}
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        response = opener.open(req, timeout=300)
        for line in response:
            if line.startswith(b'data:') and line.strip() != b'data: [DONE]':
                event = json.loads(line[5:])
                if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                    return response
        response.close()
        raise AssertionError('stream produced no token')

    with (a.output / 'server.log').open('wb') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 360
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server exited; see server.log')
                try:
                    if get('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('startup')
                time.sleep(.5)
            props = get('/props')
            evidence['props'] = props
            check('P and normalized N exposed', props['total_slots'] == a.parallel and
                  props['kvmem']['conversations'] == max(a.parallel, a.conversations) and
                  props['kvmem']['conversations_requested'] == a.conversations)
            check('physical slots remain P', len(get('/slots')) == a.parallel)
            initial = {name: ask(name, history) for name, history in histories.items()}
            expected = ({name: {'A': 'RED', 'B': 'BLUE', 'C': 'GREEN', 'D': 'YELLOW', 'E': 'WHITE'}[name]
                         for name in codes} if a.mmproj else codes)
            for name, record in initial.items():
                check('cold output isolation ' + name, expected[name] in text(record).upper())
                check('new ID is cold ' + name, hit(record) == 0)
                append_followup(name, record)
            restored = ask('A', histories['A'])
            check('restored output isolation', expected['A'] in text(restored).upper())
            check('parked conversation resumes when capacity permits',
                  (hit(restored) > 1024) if a.conversations > a.parallel and a.soft_gb == 0 else (hit(restored) == 0))
            append_followup('A', restored)
            # Invalid input must leave the full pool untouched.
            before = stats()
            status, _, _ = request('/v1/chat/completions', {'messages': histories['A'],
                'max_tokens': 32, 'cache_reset': 'invalid', 'kvmem': {'conversation_id': 'INVALID'}})
            check('invalid type rejected before queue', status == 400)
            status, _, _ = request('/v1/chat/completions', {'messages': [
                {'role': 'user', 'content': 'x ' * 12000}], 'max_tokens': 32,
                'kvmem': {'conversation_id': 'TOO_LONG'}})
            after = stats()
            check('invalid request does not switch/evict', status == 400 and
                  before['switches'] == after['switches'] and before['evictions'] == after['evictions'])

            if a.conversations > a.parallel and a.soft_gb == 0:
                # The last ID replaces A on its original lane and decodes there.
                # A must resume on another lane, replacing its GPU residency.
                last = next(reversed(histories))
                c = ask(last, histories[last])
                append_followup(last, c)
                stream = long_stream(last)
                try:
                    busy_lane = stream.headers.get('X-KVMem-Lane')
                    check('old A residency is occupied by another conversation', busy_lane == restored['lane'])
                    moved = ask('A', histories['A'])
                    check('restored A uses any idle lane', moved['lane'] != busy_lane)
                    check('moved A keeps host KV', hit(moved) > 1024 and expected['A'] in text(moved).upper())
                    append_followup('A', moved)
                finally:
                    stream.close()
                deadline = time.monotonic() + 30
                while any(s['is_processing'] for s in get('/slots')):
                    if time.monotonic() > deadline:
                        raise TimeoutError('disconnect recovery')
                    time.sleep(.05)

            # Same-ID caller waits for the stream while another ID uses the
            # idle lane. This also checks queue release after a disconnect.
            stream = long_stream('QUEUE')
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = pool.submit(ask, 'QUEUE', [{'role': 'user', 'content': 'What is 2+3? Number only.'}])
                try:
                    time.sleep(.1)
                    other = ask('INDEPENDENT', [{'role': 'user', 'content': 'What is 2+3? Number only.'}])
                    check('blocked same-ID head does not block another ID',
                          other['lane'] != stream.headers.get('X-KVMem-Lane') and text(other) == '5')
                finally:
                    stream.close()
                check('same-ID waiter recovers', text(pending.result(timeout=120)) == '5')
            if a.mmproj:
                streams = [long_stream('ENCODE_' + str(lane)) for lane in range(a.parallel)]
                try:
                    check('all lanes decoding before media preparation', all(s['is_processing'] for s in get('/slots')))
                    yellow = [{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': image_url((255, 255, 0))}},
                        {'type': 'text', 'text': 'What is the dominant color? One word only.'}]}]
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        mark = (a.output / 'server.log').read_bytes().count(b'KVMEM_VISION_PREPARED')
                        future = pool.submit(ask, 'YELLOW', yellow)
                        # Wait for a cold encode in the log while no lane
                        # can yet admit this image request.
                        encoded_while_busy = False
                        deadline = time.monotonic() + 20
                        while future.done() is False and time.monotonic() < deadline:
                            if (a.output / 'server.log').read_bytes().count(b'KVMEM_VISION_PREPARED') > mark:
                                encoded_while_busy = all(s['is_processing'] for s in get('/slots'))
                                break
                            time.sleep(.05)
                        check('projector encoded while all lanes busy', encoded_while_busy and not future.done())
                        streams[0].close()
                        result = future.result(timeout=120)
                        check('cold projector work during decode preserves image', 'YELLOW' in text(result).upper())
                finally:
                    for stream in streams:
                        stream.close()
            check('fresh admission at capacity', stats()['count'] <= max(a.parallel, a.conversations))
            if a.soft_gb:
                check('soft budget reclaims parked stores', stats()['evictions'] > 0)
            evidence['slots'] = get('/slots')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            evidence['exit_code'] = proc.returncode
            (a.output / 'results.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(evidence['checks'])}))


if __name__ == '__main__':
    main()
