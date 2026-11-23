#!/usr/bin/env python3
"""Run video-gen tests through local ComfyUI API, log time + peak VRAM/RAM.
usage: bench.py TEST [TEST...]   (tests defined in TESTS below)
"""
import json, subprocess, sys, threading, time, urllib.request, uuid, os

API = "http://127.0.0.1:8188"
D = os.path.expanduser("~/videogen")
SEED = 42

T2V = ("A golden retriever runs along the shoreline of a beach at sunset, splashing through shallow waves. "
       "The camera tracks alongside at a low angle, warm golden light, water spray sparkling in the air. "
       "The dog barks happily once; waves crash and seagulls call in the background.")
I2V = ("The cat turns its head toward the camera, blinks slowly, then walks forward through the snow toward the viewer, "
       "paws sinking into the powder, tail swaying. A gentle wind blows a few snowflakes across the frame. "
       "Soft crunching footsteps on snow, then a quiet meow.")
IMG = "Felis_catus-cat_on_snow.jpg"


def load(name):
    return json.load(open(f"{D}/api/{name}.json"))


def gguf(p, nid, f):
    p[nid] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": f}}


def drop_enhancer(p, sg):
    for n in ("380", "393", "381"):
        p.pop(f"{sg}:{n}", None)
    sw = p[f"{sg}:382"]["inputs"]
    sw["on_true"] = sw["on_false"]
    return p


def ltx_t2v(mp=0.9, dur=5):
    p = drop_enhancer(load("video_ltx2_5_t2v"), "405")
    p["405:376"]["inputs"]["value"] = T2V
    p["405:362"]["inputs"]["value"] = dur
    p["405:339"]["inputs"]["noise_seed"] = SEED
    p["409"]["inputs"]["megapixels"] = mp
    p["75"]["inputs"]["filename_prefix"] = f"bench/ltx_t2v_{mp}"
    return p


def ltx_i2v(mp=0.9, dur=5):
    p = drop_enhancer(load("video_ltx2_5_i2v"), "398")
    p["395"]["inputs"]["image"] = IMG
    p["398:376"]["inputs"]["value"] = I2V
    p["398:362"]["inputs"]["value"] = dur
    p["398:339"]["inputs"]["noise_seed"] = SEED
    p["403"]["inputs"].update(megapixels=mp, aspect_ratio="3:2 (Photo)")
    p["75"]["inputs"]["filename_prefix"] = f"bench/ltx_i2v_{mp}"
    return p


H3_LORA = "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors"


def h3(p, sg, prompt, mp, dur, ids, turbo=True):
    sw, lora, steps, clip, vae, prim_dur, noise, cond = ids
    p[f"{sg}:{sw}"]["inputs"]["value"] = turbo
    p[f"{sg}:{lora}"]["inputs"]["lora_name"] = H3_LORA
    p[f"{sg}:{steps}"]["inputs"]["value"] = 4
    p[f"{sg}:{clip}"]["inputs"]["clip_name"] = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    p[f"{sg}:{vae}"]["inputs"]["vae_name"] = "minimax_h3_video_vae_fp16.safetensors"
    p[f"{sg}:{prim_dur}"]["inputs"]["value"] = dur
    p[f"{sg}:{noise}"]["inputs"]["noise_seed"] = SEED
    p[f"{sg}:{cond}"]["inputs"]["prompt"] = prompt
    p["115"]["inputs"]["megapixels"] = mp
    return p


def h3_t2v(mp=0.4, dur=5, turbo=True):
    p = h3(load("video_minimax_h3_t2v"), "140", T2V, mp, dur, (139, 134, 138, 128, 119, 133, 129, 131), turbo)
    p["115"]["inputs"]["aspect_ratio"] = "16:9 (Widescreen)"
    p["92"]["inputs"]["filename_prefix"] = f"bench/h3_t2v_{mp}_{'turbo' if turbo else '20st'}"
    return p


def h3_i2v(mp=0.4, dur=5):
    p = h3(load("video_minimax_h3_i2v"), "105", I2V, mp, dur, (126, 121, 125, 13, 11, 111, 15, 104))
    p["114"]["inputs"]["image"] = IMG
    p["115"]["inputs"]["aspect_ratio"] = "3:2 (Photo)"
    p["92"]["inputs"]["filename_prefix"] = f"bench/h3_i2v_{mp}"
    return p


def wan_t2v(w=832, h=480, dur=5):
    p = load("video_wan2_2_14B_t2v")
    gguf(p, "128:75", "Wan2.2-T2V-A14B-HighNoise-Q5_K_M.gguf")
    gguf(p, "128:76", "Wan2.2-T2V-A14B-LowNoise-Q5_K_M.gguf")
    p["128:83"]["inputs"]["lora_name"] = "wan22_lightning_t2v_high.safetensors"
    p["128:85"]["inputs"]["lora_name"] = "wan22_lightning_t2v_low.safetensors"
    p["128:129"]["inputs"]["value"] = True
    p["128:73"]["inputs"]["vae_name"] = "Wan2.1_VAE.safetensors"
    p["128:74"]["inputs"].update(width=w, height=h)
    p["128:89"]["inputs"]["text"] = T2V
    p["128:126"]["inputs"]["value"] = dur
    p["128:81"]["inputs"]["noise_seed"] = SEED
    p["80"]["inputs"]["filename_prefix"] = f"bench/wan_t2v_{w}x{h}"
    return p


def wan_i2v(w=768, h=512, dur=5):
    p = load("video_wan2_2_14B_i2v")
    gguf(p, "129:95", "Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf")
    gguf(p, "129:96", "Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf")
    p["129:101"]["inputs"]["lora_name"] = "wan22_lightning_i2v_high.safetensors"
    p["129:102"]["inputs"]["lora_name"] = "wan22_lightning_i2v_low.safetensors"
    p["129:131"]["inputs"]["value"] = True
    p["129:90"]["inputs"]["vae_name"] = "Wan2.1_VAE.safetensors"
    p["129:98"]["inputs"].update(width=w, height=h)
    p["129:93"]["inputs"]["text"] = I2V
    p["129:161"]["inputs"]["value"] = dur
    p["129:86"]["inputs"]["noise_seed"] = SEED
    p["97"]["inputs"]["image"] = IMG
    p["108"]["inputs"]["filename_prefix"] = f"bench/wan_i2v_{w}x{h}"
    return p


TESTS = {
    "ltx_t2v": ltx_t2v, "ltx_i2v": ltx_i2v,
    "wan_t2v": wan_t2v, "wan_i2v": wan_i2v,
    "h3_t2v": h3_t2v, "h3_i2v": h3_i2v,
    "ltx_t2v_hd": lambda: ltx_t2v(mp=2.0), "h3_t2v_768": lambda: h3_t2v(mp=1.0), "h3_t2v_20st": lambda: h3_t2v(turbo=False),
    "wan_t2v_720": lambda: wan_t2v(1280, 720),
}


class Monitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.vram = 0; self.ram_min = 10**9; self.swap = 0; self.stop = False

    def run(self):
        while not self.stop:
            try:
                v = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
                self.vram = max(self.vram, v)
                mi = dict(l.split(":") for l in open("/proc/meminfo"))
                kb = lambda k: int(mi[k].split()[0]) // 1024
                self.ram_min = min(self.ram_min, kb("MemAvailable"))
                self.swap = max(self.swap, kb("SwapTotal") - kb("SwapFree"))
            except Exception:
                pass
            time.sleep(1)


def post(path, data):
    r = urllib.request.Request(API + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))


def run(name):
    p = TESTS[name]()
    m = Monitor(); m.start()
    t0 = time.time()
    try:
        pid = post("/prompt", {"prompt": p, "client_id": str(uuid.uuid4())})["prompt_id"]
    except urllib.error.HTTPError as e:
        print(name, "REJECTED", e.read().decode()[:3000]); return
    while True:
        time.sleep(2)
        h = json.load(urllib.request.urlopen(f"{API}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed") is not None or (pid in h and h[pid]["status"].get("status_str") in ("success", "error")):
            break
    dt = time.time() - t0
    m.stop = True
    st = h[pid]["status"]
    files = [f"{x.get('subfolder')}/{x['filename']}" for o in h[pid]["outputs"].values() for k in o.values() if isinstance(k, list) for x in k if isinstance(x, dict) and "filename" in x]
    err = ""
    if st.get("status_str") != "success":
        err = json.dumps([mm for mm in st.get("messages", []) if mm[0] == "execution_error"])[:2000]
    line = f"{time.strftime('%H:%M')}\t{name}\t{st.get('status_str')}\t{dt:.0f}s\tpeakVRAM={m.vram}MiB\tminAvail={m.ram_min}MiB\tswap={m.swap}MiB\t{files}"
    print(line, err, flush=True)
    open(f"{D}/bench.tsv", "a").write(line + "\n")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
