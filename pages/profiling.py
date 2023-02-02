# a function with profiles functions. i,e,. times the function and measures cpu, gpu and ram usage

from utils.utils_global import *
set_rcParams()
def func_to_profile(x):
    return np.sin(x)


# now we can profile any function
@function_profiler
def a(x):
    return func_to_profile(x)




df = pd.read_csv('profiling_data.csv', index_col=0)
with st.expander('raw data', expanded=False):
    df
df['cpu_delta'] = df['end_cpu'] - df['start_cpu']
df['ram_delta'] = df['end_ram_used'] - df['start_ram_used']
df['ram_percent_delta'] = df['end_ram_percent'] - df['start_ram_percent']
df = df[["func_name", "cpu_delta", "ram_delta", "ram_percent_delta", "run_time"]]


summean = st.radio('When we have two runs of the same function how should we groupby', ['sum', 'mean'])

if summean == 'mean': 
    df_grouped = df.groupby('func_name').mean()
else:
    df_grouped = df.groupby('func_name').sum()

figs = []
for i, col in enumerate(df_grouped.columns):
    fig, ax = plt.subplots(1,1, sharey=True, figsize=(4,4))
    try:
        lim = df_grouped[col].describe()['75%']
        
        df_grouped[df_grouped[col] > lim][col].plot.barh(ax=ax)
    except IndexError:
        pass
    plt.title(col)
    figs.append(fig)

plt.tight_layout()
plt.close()
cols = st.columns(2)
for i in range(4):
    cols[i%2].pyplot(figs[i])


caption_figure("only the values about 75th percentile are shown.")