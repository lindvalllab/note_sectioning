import matplotlib.pyplot as plt
import seaborn as sns

def annotation_overview_text(result_df, threshold, type):
    total_rows = result_df.shape[0]
    hpi_above_threshold = result_df[result_df['HPI_Interval_Hx'] >= threshold].shape[0]
    a_p_above_threshold = result_df[result_df['A&P'] >= threshold].shape[0]
    both_above_threshold = result_df[(result_df['HPI_Interval_Hx'] >= threshold) & (result_df['A&P'] >= threshold)].shape[0]
    history_interval_hx_percentage = (hpi_above_threshold / total_rows) * 100
    a_p_percentage = (a_p_above_threshold / total_rows) * 100
    both_percentage = (both_above_threshold / total_rows) * 100
    message = (
        f"For {type} ({total_rows} samples):\n"
        f"- {hpi_above_threshold} samples ({history_interval_hx_percentage:.2f}%) have 'HPI_Interval_Hx' score greater than {threshold}.\n"
        f"- {a_p_above_threshold} samples ({a_p_percentage:.2f}%) have 'A&P' score greater than {threshold}.\n"
        f"- {both_above_threshold} samples ({both_percentage:.2f}%) have both scores greater than {threshold}."
    )
    print(message)

def annotation_overview_plot(result_df, threshold, type):
    # Use a built-in modern style
    plt.style.use('ggplot')

    # Customize the color palette
    sns.set_palette('muted')

    # Assuming you already have 'result_df' defined
    hpi_scores = result_df['HPI_Interval_Hx']
    ap_scores = result_df['A&P']

    # Create two histograms with enhanced appearance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    fig.suptitle(f"{type}", fontsize=16, weight='bold')

    # Histogram for HPI_Interval_Hx scores with a more modern aesthetic
    sns.histplot(hpi_scores, bins=20, kde=True, color='dodgerblue', ax=ax1)
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel("HPI_Interval_Hx Score", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

    # Histogram for A&P scores with enhanced appearance
    sns.histplot(ap_scores, bins=20, kde=True, color='mediumseagreen', ax=ax2)
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("A&P Score", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

    # Adjusting the layout for a more balanced look
    plt.tight_layout()

    # Show the histograms
    plt.show()