"""Horizontal bar plot of the number of drugs tested in each cohort."""

from plot_dataset_summary import load_summary, plot_metric_barplot


def main():
    df = load_summary()
    plot_metric_barplot(df, "Drug", xlabel="Drug", out_name="barplot_drugs", color="#F6BEC0")


if __name__ == "__main__":
    main()
