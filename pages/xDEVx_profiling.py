# a function with profiles functions. i,e,. times the function and measures cpu, gpu and ram usage

from utils.utils_global import *
set_rcParams()

'# Performance info'

try:
    df = pd.read_csv('profiling_data.csv', index_col=0)
    with st.expander('raw data', expanded=False):
        df
        pass
except:
    df = pd.DataFrame()
    pass


df['cpu_delta'] = df['end_cpu'] - df['start_cpu']
df['ram_delta'] = df['end_ram_used'] - df['start_ram_used']
df['ram_percent_delta'] = df['end_ram_percent'] - df['start_ram_percent']
df = df[["func_name", "cpu_delta", "ram_delta", "ram_percent_delta", "run_time"]]

cols = st.columns((1,1))
deviations = cols[1].slider('Show those more than x sigmas from the mean', 0.,1.2,.2)
if devmod:
    @function_profiler
    def a(x):
        return 3*(x)

    if cols[1].button('test'):
        a(4)

df_grouped = {
    'mean' : df.groupby('func_name').mean(),
    'sum'  : df.groupby('func_name').sum(),
    'max'  : df.groupby('func_name').max(),
    'count'  : df.groupby('func_name').count(),
}[cols[0].radio('How should mutiple calls of a func be grouped?', [ 'max', 'count','sum', 'mean',])]


figs = {}
for i, col in enumerate(df_grouped.columns):
    fig, ax = plt.subplots(1,1, sharey=True, figsize=(4,4))
    try:
        mu = df_grouped[col].mean()
        sig = df_grouped[col].std()
        lim = mu +deviations*sig 
        
        df_grouped[df_grouped[col] >= lim][col].plot.barh(ax=ax)
    except IndexError:
        pass
    plt.title(col)
    figs[col] = fig

plt.tight_layout()
plt.close()
cols = st.columns(2)
for i, col in enumerate(figs):
    cols[i%2].pyplot(figs[col])
    caption_figure(f"{col}", st=cols[i%2])
    def vspace(n=1, st=st):
        for i in range(n):
            st.markdown('')
    vspace(n=2, st=cols[i%2])
caption_figure(f"Only showing values greater than {deviations} " +  r"$\sigma$.", st=st)

'---'

"""
*note*

To collect more date, run locally with devmod enabled. devmod can be toggled in utils_global.


---

It seems figure numbers keep going up. Are we keeping all these figs in ram?
"""