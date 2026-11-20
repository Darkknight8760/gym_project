"""
Project 12 - Gym & Fitness Center Membership Analytics
=======================================================
Run:  python gym_analysis.py  (needs pandas, numpy, matplotlib)
Input : 12_gym_membership.csv (same folder)
Output: ./outputs/  -> cleaned data, summary tables, PNG charts, dashboard.html
Open outputs/dashboard.html in any browser for the interactive version.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw charts to files, no window needed
import matplotlib.pyplot as plt

OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------- Task 1: load & inspect
df = pd.read_csv("12_gym_membership.csv")
print("Shape:", df.shape)
print(df.dtypes, "\n")
print(df.head(), "\n")

# ---------------------------------------------------------------- Task 2: missing + duplicates
print("Missing values per column:\n", df.isna().sum(), "\n")
dup_ids = df["Checkin_ID"].duplicated().sum()
print("Duplicate Checkin_ID rows:", dup_ids)
# Are the duplicates exact copies? (if yes, dropping is safe - no information is lost)
print("Exact duplicate rows     :", df.duplicated().sum(), "\n")

# ---------------------------------------------------------------- Task 3: cleaning
df = df.drop_duplicates(subset="Checkin_ID", keep="first")   # one row per check-in

# Missing Fitness_Class = member used the gym floor without booking a class.
# That is a real behaviour, so we label it instead of deleting 18% of the data.
df["Fitness_Class"] = df["Fitness_Class"].fillna("No Class (Open Gym)")

# Missing duration: impute with the MEDIAN duration of the same class.
# Median (not mean) because it is robust to very long/short sessions, and the
# class-level median keeps e.g. HIIT and Yoga sessions realistic.
n_missing = df["Workout_Duration_Mins"].isna().sum()
df["Duration_Imputed"] = df["Workout_Duration_Mins"].isna()  # keep a flag for transparency
df["Workout_Duration_Mins"] = df["Workout_Duration_Mins"].fillna(
    df.groupby("Fitness_Class")["Workout_Duration_Mins"].transform("median"))
print(f"Imputed {n_missing} durations with class-level median.\n")

# ---------------------------------------------------------------- Task 4: datetime features
df["Checkin_Time"] = pd.to_datetime(df["Checkin_Time"])
df["Weekday"] = df["Checkin_Time"].dt.day_name()
df["Hour"] = df["Checkin_Time"].dt.hour
df["Month"] = df["Checkin_Time"].dt.to_period("M").astype(str)
df["Is_Weekend"] = df["Weekday"].isin(["Saturday", "Sunday"]
)
# ---------------------------------------------------------------- Task 5: headline numbers
print("Total check-ins       :", len(df))
print("Unique members        :", df["Member_ID"].nunique())
print("Avg workout duration  : %.1f min\n" % df["Workout_Duration_Mins"].mean())

# ---------------------------------------------------------------- Task 6: tier comparison
tier = df.groupby("Membership_Tier").agg(
    Checkins=("Checkin_ID", "count"),
    Members=("Member_ID", "nunique"),
    Avg_Duration=("Workout_Duration_Mins", "mean"),
    Median_Duration=("Workout_Duration_Mins", "median"))
tier["Visits_per_Member"] = tier["Checkins"] / tier["Members"]   # fair comparison across tier sizes
tier = tier.round(2).loc[["Basic", "Gold", "Platinum"]]
print("Tier comparison:\n", tier, "\n")

# ---------------------------------------------------------------- Task 7: peak hours & busiest days
by_day = df["Weekday"].value_counts().reindex(DAYS)
by_hour = df["Hour"].value_counts().sort_index()
heat = df.pivot_table(index="Weekday", columns="Hour", values="Checkin_ID",
                      aggfunc="count", fill_value=0).reindex(DAYS)   # day x hour grid
top_slots = heat.stack().sort_values(ascending=False).head(5)
print("Busiest day:", by_day.idxmax(), "| Top 3 hours:", by_hour.nlargest(3).index.tolist())
print("Top 5 day-hour slots:\n", top_slots, "\n")

# ---------------------------------------------------------------- Task 8: class ranking
classes = df["Fitness_Class"].value_counts().rename("Attendance").to_frame()
classes["Share_%"] = (classes["Attendance"] / len(df) * 100).round(1)
classes["Avg_Duration"] = df.groupby("Fitness_Class")["Workout_Duration_Mins"].mean().round(1)
print("Class ranking:\n", classes, "\n")

# ---------------------------------------------------------------- Task 9: gender comparison
gender = df.groupby("Gender")["Workout_Duration_Mins"].agg(["count", "mean", "median"]).round(2)
print("Duration by gender:\n", gender, "\n")

# ---------------------------------------------------------------- Task 10: monthly revenue by tier
# DATA QUALITY NOTE: Monthly_Fee is NOT tied to Membership_Tier in this file (e.g. Basic rows
# carry 999, 1499 and 2499). A fee is a monthly charge, so we must not sum it per check-in
# (a member who visits 8 times is still billed once). We therefore bill each
# (Member, Month, Tier) once, using the fee recorded on that member's first check-in of the month.
bill = (df.sort_values("Checkin_Time")
          .groupby(["Month", "Membership_Tier", "Member_ID"], as_index=False)["Monthly_Fee"].first())
revenue = bill.pivot_table(index="Month", columns="Membership_Tier", values="Monthly_Fee",
                           aggfunc="sum", fill_value=0)
revenue["Total"] = revenue.sum(axis=1)
print("Revenue by tier (last 6 months):\n", revenue.tail(6), "\n")
print("Fee values seen per tier:\n", df.groupby("Membership_Tier")["Monthly_Fee"].unique(), "\n")

# ---------------------------------------------------------------- Task 11: low-engagement members
# Rule: fewer than 2 visits per month, measured over the member's own ACTIVE months
# (months in which they appeared at least once) so a new joiner is not unfairly penalised.
mm = df.groupby(["Member_ID", "Month"]).size().rename("v").reset_index()
member = mm.groupby("Member_ID").agg(Active_Months=("Month", "nunique"), Visits=("v", "sum"))
member["Visits_per_Active_Month"] = member["Visits"] / member["Active_Months"]
member["Low_Engagement"] = member["Visits_per_Active_Month"] < 2
print("Low-engagement members:", member["Low_Engagement"].sum(), "of", len(member), "\n")

# ---------------------------------------------------------------- Task 12: NumPy statistics
d = df["Workout_Duration_Mins"].to_numpy()
q1, med, q3 = np.percentile(d, [25, 50, 75])
print(f"Median={med:.1f}  Variance={np.var(d, ddof=1):.1f}  Std={np.std(d, ddof=1):.1f}")
print(f"Quartiles: Q1={q1:.1f}, Q2={med:.1f}, Q3={q3:.1f}\n")

# ---------------------------------------------------------------- Task 13: IQR outliers
iqr = q3 - q1
low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["Outlier"] = np.where(d < low_fence, "Unusually short",
                 np.where(d > high_fence, "Unusually long", "Normal"))
print(f"IQR fences: {low_fence:.1f} to {high_fence:.1f} -> outliers:\n", df["Outlier"].value_counts(), "\n")

# ---------------------------------------------------------------- Task 14: intensity labels
# Documented thresholds: Short < 45 min, Moderate 45-89 min, Extended >= 90 min
# (45 and 90 sit close to the 25th / 75th percentiles, so each band is meaningful).
df["Intensity"] = np.select([d < 45, d < 90], ["Short", "Moderate"], default="Extended")
print(df["Intensity"].value_counts(), "\n")

# ================================================================ EXTRA ANALYTICS
# E1. Member segmentation (RFM-style): Recency / Frequency / Duration
end = df["Checkin_Time"].max()
rfm = df.groupby("Member_ID").agg(Last=("Checkin_Time", "max"), Freq=("Checkin_ID", "count"),
                                  AvgMin=("Workout_Duration_Mins", "mean"))
rfm["Days_Since_Last"] = (end - rfm["Last"]).dt.days
rfm["Segment"] = np.select(
    [rfm["Days_Since_Last"] > 180,
     (rfm["Days_Since_Last" ]<= 60) & (rfm["Freq"] >= 4),
     rfm["Days_Since_Last"] <= 90],
    ["Churn risk (180d+ silent)", "Champion", "Active"], default="Cooling off")
print("Segments:\n", rfm["Segment"].value_counts(), "\n")

# E2. Class x time-of-day mix (which class owns which part of the day?)
df["Daypart"] = pd.cut(df["Hour"], [4, 11, 16, 24], labels=["Morning", "Afternoon", "Evening"])
class_daypart = pd.crosstab(df["Fitness_Class"], df["Daypart"])

# E3. Weekday vs weekend and year-over-year check-ins
yoy = df.groupby([df["Checkin_Time"].dt.year, df["Checkin_Time"].dt.month]).size().unstack(0)

# E4. Statistical test: is the gender difference in duration real or noise? (Welch t-stat via NumPy)
g = [df.loc[df["Gender"] == k, "Workout_Duration_Mins"].to_numpy() for k in ["Female", "Male"]
]
t = (g[0].mean() - g[1].mean()) / np.sqrt(g[0].var(ddof=1) / len(g[0]) + g[1].var(ddof=1) / len(g[1]))
print(f"Female vs Male duration: Welch t = {t:.2f} (|t| < 2 => difference not significant)\n")

# ---------------------------------------------------------------- Task 15: recommendations
real = classes.drop(index="No Class (Open Gym)")     # recommend only actual scheduled classes
one_visit = int((member["Visits"] == 1).sum())          # members who came exactly once: sharper target than the <2 rule
recs = [
    f"Staff and schedule up for {by_day.idxmax()} and the {by_hour.idxmax()}:00 hour, the busiest slots; shift cleaning/maintenance to the quietest hours ({by_hour.idxmin()}:00).",
    f"Add sessions for '{real.index[0]}' and '{real.index[1]}' (top attendance); review or re-time '{real.index[-1]}', the least attended class.",
    f"{int(member['Low_Engagement'].sum())} of {len(member)} members meet the '<2 visits/month' rule because the data is sparse; prioritise the {one_visit} who visited only once with a welcome-back offer, and run a 30-day streak challenge for the rest.",
    f"{(rfm['Segment'].str.startswith('Churn')).sum()} members have been silent 180+ days: send a win-back offer (free PT session) before cancellation.",
    "Fix billing data: Monthly_Fee does not match Membership_Tier, so revenue cannot be trusted until fees are standardised per tier.",
]
print("RECOMMENDATIONS"); [print(f" {i+1}. {r}") for i, r in enumerate(recs)]

# ---------------------------------------------------------------- save tables
df.to_csv(f"{OUT}/cleaned_checkins.csv", index=False)
for name, tbl in {"tier": tier, "classes": classes, "gender": gender, "revenue": revenue,
                  "heatmap": heat, "members": member.join(rfm[["Segment", "Days_Since_Last"]])}.items():
    tbl.to_csv(f"{OUT}/table_{name}.csv")
open(f"{OUT}/recommendations.txt", "w").write("\n".join(f"{i+1}. {r}" for i, r in enumerate(recs)))

# ---------------------------------------------------------------- static charts
fig, ax = plt.subplots(2, 3, figsize=(17, 9)); ax = ax.ravel()
im = ax[0].imshow(heat.values, aspect="auto", cmap="YlOrRd")
ax[0].set_yticks(range(7), [x[:3] for x in DAYS]); ax[0].set_xticks(range(len(heat.columns)), heat.columns, fontsize=7)
ax[0].set_title("Check-in heatmap (day x hour)"); fig.colorbar(im, ax=ax[0])
by_day.plot.bar(ax=ax[1], color="#2a9d8f", title="Check-ins by weekday")
classes["Attendance"].plot.barh(ax=ax[2], color="#e76f51", title="Class attendance").invert_yaxis()
tier["Visits_per_Member"].plot.bar(ax=ax[3], color="#264653", title="Visits per member by tier")
revenue.drop(columns="Total").plot(ax=ax[4], title="Monthly revenue by tier")
ax[5].boxplot(d, vert=False); ax[5].set_title("Workout duration (IQR outlier view)")
plt.tight_layout(); plt.savefig(f"{OUT}/charts.png", dpi=120)

# ---------------------------------------------------------------- interactive dashboard
# Ensure template layout exists before opening
tpl_fallback = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Gym & Fitness Center Analytics Dashboard</title>
<style>
  body { font-family: sans-serif; margin: 20px; background: #f4f6f9; color: #333; }
  h1 { color: #111; }
  .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  ul { line-height: 1.6; }
</style>
</head>
<body>
<div class="container">
  <h1>Gym Membership & Analytics Dashboard</h1>
  <h3>Key Actionable Recommendations:</h3>
  <ul id="recs-list"></ul>
</div>
<script>
  const data = __DATA__;
  const recs = __RECS__;
  const list = document.getElementById('recs-list');
  recs.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    list.appendChild(li);
  });
</script>
</body>
</html>"""
if not os.path.exists("dashboard_template.html"):
    with open("dashboard_template.html", "w", encoding="utf-8") as f:
        f.write(tpl_fallback)

# Embed the cleaned rows as JSON into the HTML template; all filtering runs in the browser.
rows = df[["Checkin_ID", "Member_ID", "Membership_Tier", "Gender", "Fitness_Class", "Weekday", "Hour",
           "Month", "Workout_Duration_Mins", "Monthly_Fee"]].copy()
rows["Workout_Duration_Mins"] = rows["Workout_Duration_Mins"].round(1)
tpl = open("dashboard_template.html", encoding="utf-8").read()
open(f"{OUT}/dashboard.html", "w", encoding="utf-8").write(
    tpl.replace("__DATA__", json.dumps(rows.to_dict("records"))).replace("__RECS__", json.dumps(recs)))
print("\nDone. See the outputs/ folder.")