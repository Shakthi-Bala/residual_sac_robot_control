## UR16e: conventional control, SAC, and residual SAC

Matched simulation rollouts, shown side by side: **Cartesian PI · standalone SAC · residual SAC**. Red marks the target. Each uses training seed 1 and target seed 20000, fixed before evaluating results; these are illustrations rather than averages. [Protocol, metrics, and limitations](residual-control/README.md).

### Nominal reaching

![UR16e nominal reaching: three matched controllers](residual-control/results/media/nominal_reach.gif)

### Reaching with 3 kg added payload and 100 ms command delay

![UR16e perturbed reaching](residual-control/results/media/combined_reach.gif)

### Tracking with 3 kg added payload and 100 ms command delay

![UR16e perturbed tracking](residual-control/results/media/combined_tracking.gif)

### Tracking beyond the training payload range: 5 kg and 100 ms delay

![UR16e held-out payload tracking](residual-control/results/media/ood_payload_tracking.gif)

### Tracking beyond the training delay range: 3 kg and 160 ms delay

![UR16e held-out delay tracking](residual-control/results/media/ood_delay_tracking.gif)

