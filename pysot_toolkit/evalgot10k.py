from got10k.experiments import ExperimentGOT10k

e = ExperimentGOT10k(
    root_dir=''  # Set your GOT-10k val dataset path here,
    subset='val'   # ⚠️ 一定用 val
)

e.report(['sot'])