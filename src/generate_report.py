import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
df = pd.read_csv('data/squid_dataset.csv')

# Create the profile report
profile = ProfileReport(df, title="Squid Data Analysis Report", explorative=True)

# Save the report to an HTML file
profile.to_file("squid_data_analysis_report.html")

print("✅ Analysis report created: 'squid_data_analysis_report.html'")