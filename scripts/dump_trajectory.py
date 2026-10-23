import csv, sys

for f in sys.argv[1:]:
    print("==", f)
    rows = list(csv.DictReader(open(f)))
    header = ("step", "res_max", "res_mean", "ics_mean", "bcs_mean",
              "ics_w", "bcs_w", "total_loss", "p_max")
    print("{:>5} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>11} {:>8}".format(*header))
    for r in rows:
        vals = (
            r["step"],
            "{:.3e}".format(float(r["res_max_abs_grad"])),
            "{:.3e}".format(float(r["res_mean_abs_grad"])),
            "{:.3e}".format(float(r["ics_mean_abs_grad"])),
            "{:.3e}".format(float(r["bcs_mean_abs_grad"])),
            "{:.3e}".format(float(r["ics_weight_new"])),
            "{:.3e}".format(float(r["bcs_weight_new"])),
            "{:.3e}".format(float(r["total_loss"])),
            "{:.3f}".format(float(r["param_max_abs"])),
        )
        print("{:>5} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11} {:>11} {:>8}".format(*vals))
