# src/sigbandit/tests/hospital_staff/load_data.py

import pandas as pd
import matplotlib.pyplot as plt
import os


def load_nyc_vomit_counts(
	csv_path: str,
	*,
	date_col: str = "Date",
	count_col: str = "Count",
	borough_col: str = "Borough",
	age_col: str = "Age Group",
	citywide_label: str = "Citywide",
    all_age_label: str = "All age groups",
	trim: bool = True,
):
	"""
	Load NYC EpiQuery vomiting counts export.

	Expected CSV columns (like your screenshot):
	  Borough, Age Group, Date, Count

	Returns:
	  df: tidy DataFrame with parsed Date and numeric Count
	  boroughs: sorted unique borough values
	  age_groups: sorted unique age-group values
	"""
	df = pd.read_csv(csv_path)

	# Clean up strings
	df[borough_col] = df[borough_col].astype(str).str.strip()
	df[age_col] = df[age_col].astype(str).str.strip()

	# Parse date
	df[date_col] = pd.to_datetime(df[date_col].astype(str).str.strip(), format="%m/%d/%y", errors="coerce")
	df = df.dropna(subset=[date_col])

	# Count -> numeric
	df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
	df = df.dropna(subset=[count_col])

	# Filter to Citywide + All ages
	df_cw = df[(df[borough_col] == citywide_label) & (df[age_col] == all_age_label)].copy()
	df_cw = (
        df_cw.groupby(date_col, as_index=False)[count_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

	if trim:
		start = df_cw[date_col].min().normalize()

		# days to add to reach desired weekday
		monday = 0
		delta = (monday - start.weekday()) % 7
		first_sunday = start + pd.Timedelta(days=int(delta))

		df_cw = df_cw[df_cw[date_col] >= first_sunday].reset_index(drop=True)

		# end on sunday
		n = len(df_cw)
		n_full = (n // 7) * 7
		df_cw = df_cw.iloc[:n_full].reset_index(drop=True)

	return df_cw
	

def plot_vomit_counts(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    count_col: str = "Count",
    save_path: str,
    title: str = "NYC Vomiting ED Visits",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[date_col], df[count_col], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



# test
if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--csv", type=str, default="tests/hospital_staff/nyc_vomit_count.csv")
	args = parser.parse_args()

	df = load_nyc_vomit_counts(args.csv)

	print(f"Loaded df: shape={df.shape}")
	print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

	# quick peek
	print("\nHEAD:")
	print(df.head(10))

	# plot
	plot_vomit_counts(df, save_path="experiments/figs/hospital_staff/nyc_vomit_counts.png")