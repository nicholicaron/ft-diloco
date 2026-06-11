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

```bash
# find cheap interruptible 4090s, distinct geographies
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>200 reliability>0.95' \
    --interruptible -o 'dph+' | head -15
# rent (PRICE = bid; instance pauses if outbid — that's a fault event for us)
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime \
    --disk 30 --bid <PRICE> \
    --env TS_AUTHKEY=$(cat ~/.config/ftd_ts_authkey) \
    --env REPLICA_ID=2 --env N_REPLICAS=3 \
    --env LIGHTHOUSE=http://100.86.208.63:29510 \
    --onstart scripts/cloud/bootstrap_vast.sh
vastai show instances
vastai destroy instance <ID>          # ALWAYS destroy (not stop) when done — stopped
                                      # instances keep billing storage
```

worker4 joins the same run from home with `FTD_ADVERTISE_HOST=$(tailscale ip -4)`,
`GLOO_SOCKET_IFNAME=tailscale0`, lighthouse via worker1's TS IP.

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
