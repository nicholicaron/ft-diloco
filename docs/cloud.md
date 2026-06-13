# M4 cloud hybrid: worker4 + Vast.ai spot 4090s over Tailscale

Topology: worker4 (RTX 3060, home) + 1–2 Vast.ai **interruptible** RTX 4090s in other
regions, all on the Tailscale mesh; lighthouse stays on worker1. Real cross-region WAN,
real spot preemptions (= organic fault injection). Budget: $50 first tranche, hard cap;
a full session costs ~$10–15.

## Why these choices

- **Interruptible (~$0.18/hr)** over on-demand: 2–3× cheaper AND preemptions are data.
- **Tailscale mesh**: flat 100.x addressing means `FTD_ADVERTISE_HOST`/`MASTER_ADDR`
  machinery works unchanged behind both home NAT and Vast's docker networking. DiLoCo's
  rare syncs tolerate DERP relaying if direct paths fail.
- Known risk (validate with the $5 smoke test first): Vast containers without
  `/dev/net/tun` fall back to tailscaled userspace netstack; inbound TCP to the TS IP
  is proxied — gloo/HTTP transports *should* work but this is the first thing to verify.
  If it fails: re-rent filtering for hosts that expose tun.

## One-time setup (human)

1. Vast.ai account at https://cloud.vast.ai → add **$25–50** credit.
2. API key (Account → API Key) → on the Mac: `echo '<key>' > ~/.vast_api_key && chmod 600 ~/.vast_api_key`
3. Tailscale auth key: https://login.tailscale.com/admin/settings/keys →
   **Reusable + Ephemeral** (ephemeral nodes vanish from the tailnet when destroyed) →
   `echo '<tskey-...>' > ~/.config/ftd_ts_authkey && chmod 600 ~/.config/ftd_ts_authkey`
4. `pip/uv install vastai` (CLI) — agent handles the rest.

## Operations (agent-driven, for reference)

**MUST use a real VM, not a docker container.** Docker instances only get tailscale
*userspace-netstack*, which this stack cannot use: the TCPStore can't bind the TS
address, gloo can't dial it, recovery URLs are unreachable. Launch a true KVM VM.

```bash
# 1. find VM-CAPABLE 4090s (verified hosts; on-demand is more reliable than
#    interruptible for the launch itself — preemption is injected later, see below)
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>200 reliability>0.98 \
    disk_space>40 vms_enabled=true' -o 'dph+'
# 2. rent via the OFFICIAL VM template (image=docker.io/vastai/kvm). Account SSH key
#    must be registered first: vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"
vastai create instance <OFFER_ID> \
    --template_hash b7942f6bbc4374893ff66eb78145bbac \
    --disk 40 \
    --env "-e TS_AUTHKEY=$(cat ~/.config/ftd_ts_authkey) -e REPLICA_ID=1 \
           -e N_REPLICAS=2 -e LIGHTHOUSE=http://100.86.208.63:29510 \
           -e PUBKEY='$(cat ~/.ssh/id_rsa.pub)'" \
    --onstart scripts/cloud/bootstrap_vast.sh
# 3. SSH IN VIA THE DIRECT PUBLIC IP + MAPPED PORT, not the ssh proxy (proxy refuses
#    on VMs). Get them from the ports map:
vastai show instance <ID> --raw | python3 -c \
  'import json,sys; d=json.load(sys.stdin); \
   print(d["public_ipaddr"], d["ports"]["22/tcp"][0]["HostPort"])'
ssh -i ~/.ssh/id_rsa -p <MAPPED_PORT> root@<PUBLIC_IP>
vastai destroy instance <ID>   # ALWAYS destroy, never stop (stopped pods bill storage)
```

worker4 joins from home: `bash scripts/run_m4_cloud_w4.sh m4-cloud <N> <STEPS>`
(advertises its tailscale0 IP; lighthouse via worker1's TS IP).

### Hard-won gotchas (all hit during the $5 smoke, all now handled in bootstrap)
- **2FA**: account needs 2FA enabled before the API can create instances (401 otherwise).
- **`vms_enabled=true` is not enough** — a plain `--image ubuntu:22.04` on a VM-capable
  host still gives a *docker container* (no /dev/net/tun, PID 1 ≠ systemd). Only the
  `vastai/kvm:*` images / VM template produce a real VM.
- **authorized_keys perms bug**: VM images ship `/root/.ssh/authorized_keys` with modes
  sshd refuses (`bad ownership or modes`) → locked out. bootstrap chmods it [0/6]; if
  you're already locked out, fix via the **web console** at cloud.vast.ai/instances.
- **SSH proxy (`sshN.vast.ai:port`) is refused on VMs** — use direct public IP + the
  `ports["22/tcp"]` HostPort instead.
- **VM CLI quirks**: `--env` takes ONE quoted `-e K=V -e K2=V2` string, not repeated
  flags; create sometimes returns `success:false` but the instance still materializes
  (check `show instances-v1`); raw JSON output is wrapped in a Rich box (regex the array).

## Run plan ($50 tranche)

1. **$5 smoke**: 1 cheapest interruptible 4090, any region → bootstrap → verify
   tailscale reachability + 2-replica sync with worker4 → destroy. Validates the
   userspace-netstack risk before real spend.
2. **$10–15 headline**: 2× 4090 (different regions, e.g. US-east + EU) + worker4 =
   3 replica groups, H=100, tiny50m, ~1–2 h: convergence + per-sync wall time vs the
   netem prediction + comm volume. Preemptions logged as organic faults.
3. **Optional repeat** with H sweep or a kill demo on a cloud node (the GIF, but the
   dying machine is a real rented GPU in another country).

Cost ledger lives in this file as runs happen (brief open question #3: measured, not
estimated).

## Ledger

| date | what | $ |
|---|---|---|
| 2026-06-12 | smoke test (incl. all false-start instances + the passing Virginia 4090 VM) | 1.86 |
| 2026-06-12/13 | headline: base124m, worker4 + Virginia/Iceland 4090 VMs over WAN | 1.26 |
| | **total cloud spend** | **3.12** |

**Smoke test result (PASSED):** worker4 (home, RTX 3060 GPU) + a Virginia RTX 4090 VM
trained as one DiLoCo cluster over the real internet (tailscale mesh, ~52 ms RTT).
First sync at step 100 committed with **2 participants**, and the model param digests
were **bit-identical across the internet** (`a573c3de001da30a` on both). The 204 MB
fp32 pseudo-gradient allreduce traversed the WAN cleanly. $1.86 of the $5 smoke budget.

**Headline result:** base124m (GPT-2-small) DiLoCo across worker4 (home 3060) + Virginia
RTX 4090, over tailscale. Both nodes converged **~11 → ~2.2 eval loss** over 26 M tokens
on the real internet (plot `plots/m4_cloud.png`). Surfaced two coordination findings
(see findings-171.md): (1) with `min_replica_size=1` + unsynchronized or
heterogeneous-speed nodes, DiLoCo silently degrades to independent solo runs (sync points
never align); (2) the `min_replica_size>=2` barrier that forces true averaging OOM-killed
the 3060 host before the first barrier sync — needs investigation. The fully-automated
VM onstart bootstrap (key-fix → tailscale → deps → data → launch) works end-to-end. Total
cloud spend across smoke + headline: **$3.12** of the $50 tranche.
