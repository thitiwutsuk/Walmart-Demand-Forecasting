"""Dashboard สรุปผลสำหรับผู้บริหาร: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import PROCESSED, RAW, load_long

ROOT = PROCESSED.parents[1]
ANALYSIS = ROOT / 'analysis'
DAILY_PATH = PROCESSED / 'dashboard_daily.parquet'
FORECAST_PATH = PROCESSED / 'forecast_next_28_days.parquet'
STATES = ['CA', 'TX', 'WI']
CATS = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

st.set_page_config(page_title='Walmart Sales Dashboard', page_icon='📈', layout='wide')


@st.cache_data
def load_daily():
    """ยอดขายรายวัน × สาขา × หมวด (ประมาณ 6 หมื่นแถว แทน 60 ล้านแถว) สร้างครั้งแรกแล้วเก็บไว้"""
    if not DAILY_PATH.exists():
        long = load_long()
        long = long[long['sales'].notna()]
        long['revenue'] = long['sales'] * long['sell_price'].fillna(0)
        daily = (long.groupby(['d', 'state_id', 'store_id', 'cat_id'], observed=True)[['sales', 'revenue']]
                 .sum().reset_index())
        daily.to_parquet(DAILY_PATH, index=False)
    daily = pd.read_parquet(DAILY_PATH)
    cal = pd.read_csv(RAW / 'calendar.csv', usecols=['d', 'date', 'event_name_1', *[f'snap_{s}' for s in STATES]])
    cal['d'] = cal['d'].str[2:].astype(int)
    daily = daily.merge(cal, on='d')
    daily['date'] = pd.to_datetime(daily['date'])
    daily['snap'] = 0
    for s in STATES:
        daily.loc[daily['state_id'] == s, 'snap'] = daily[f'snap_{s}']
    for c in ['state_id', 'store_id', 'cat_id']:
        daily[c] = daily[c].astype(str)
    return daily


def read_csv(name, **kw):
    path = ANALYSIS / name
    return pd.read_csv(path, **kw) if path.exists() else None


# ---------- ตัวกรอง ----------
st.sidebar.header('ตัวกรอง')
states = st.sidebar.multiselect('รัฐ', STATES, STATES)
cats = st.sidebar.multiselect('หมวดสินค้า', CATS, CATS)
st.sidebar.caption('ข้อมูล: Walmart M5 (Kaggle) 10 สาขา, 3,049 สินค้า, 29 ม.ค. 2011 – 22 พ.ค. 2016')

daily = load_daily()
df = daily[daily['state_id'].isin(states) & daily['cat_id'].isin(cats)]
if df.empty:
    st.warning('เลือกรัฐและหมวดอย่างน้อยอย่างละ 1 รายการ')
    st.stop()

models = read_csv('model_comparison.csv')
impact = read_csv('business_impact.csv', index_col=0)

# ---------- หัวข้อ + KPI ----------
st.title('📈 Walmart: พยากรณ์ยอดขาย 28 วันล่วงหน้า')
st.markdown('**คำถามธุรกิจ:** ถ้ารู้ล่วงหน้าว่าจะขายได้เท่าไร จะสั่งของได้พอดี '
            'ลดของขาดชั้นวาง (เสียยอดขาย) และลดของล้นสต็อก (เงินจม / ของเสีย)')

last_d = df['d'].max()
last_year = df[df['d'] > last_d - 365]
first_year = df[df['d'] <= df['d'].min() + 364]
growth = last_year['sales'].sum() / first_year['sales'].sum() - 1

k = st.columns(4)
k[0].metric('รายได้ 12 เดือนล่าสุด', f"${last_year['revenue'].sum() / 1e6:,.1f}M")
k[1].metric('ยอดขายเติบโต (ปีแรก → ปีล่าสุด)', f'{growth:+.0%}')
if models is not None:
    best = models.iloc[0]
    base = models.set_index('method').loc['Seasonal naive', 'WRMSSE']
    k[2].metric(f"ความคลาดเคลื่อนลดลง ({best['method']})", f"{1 - best['WRMSSE'] / base:.0%}",
                help='เทียบ WRMSSE กับวิธีเดิม (ใช้ยอดสัปดาห์ก่อนเป็นค่าทาย) ยิ่งต่ำยิ่งแม่น')
if impact is not None:
    total = impact.loc['ALL'] if 'ALL' in impact.index else impact.sum()
    k[3].metric('ประหยัดได้ต่อปี (10 สาขา)', f"${total['per year, 10 stores'] / 1e6:,.2f}M",
                help=f"ถ้าขยายทุกสาขา (ประมาณคร่าวๆ): ${total['per year, all stores (rough)'] / 1e9:,.2f}B ต่อปี")
else:
    k[3].metric('ประหยัดได้ต่อปี', 'รอผล notebook 03')

# ---------- 1. ยอดขายมาจากไหน ----------
st.header('1. ยอดขายมาจากไหน')
weekly = df.set_index('date').resample('W')['revenue'].sum().iloc[1:-1].reset_index()
fig = px.line(weekly, x='date', y='revenue', labels={'date': '', 'revenue': 'รายได้ต่อสัปดาห์ ($)'})
fig.update_layout(title='แนวโน้มรายได้รายสัปดาห์', height=320, margin=dict(t=40, b=10))
st.plotly_chart(fig, width='stretch')

c1, c2 = st.columns(2)
by_store = last_year.groupby(['store_id', 'cat_id'])['revenue'].sum().reset_index()
fig = px.bar(by_store, x='store_id', y='revenue', color='cat_id', category_orders={'cat_id': CATS},
             labels={'store_id': 'สาขา', 'revenue': 'รายได้ 12 เดือน ($)', 'cat_id': 'หมวด'})
fig.update_layout(title='รายได้ตามสาขาและหมวด (12 เดือนล่าสุด)', height=360, margin=dict(t=40, b=10))
c1.plotly_chart(fig, width='stretch')

share = last_year.groupby('cat_id')['revenue'].sum().reindex(CATS).dropna()
fig = px.pie(values=share.values, names=share.index, hole=0.5)
fig.update_layout(title='สัดส่วนรายได้ตามหมวด', height=360, margin=dict(t=40, b=10))
c2.plotly_chart(fig, width='stretch')

# ---------- 2. อะไรทำให้ยอดขายขึ้นลง ----------
st.header('2. อะไรทำให้ยอดขายขึ้นลง')
total_day = df.groupby('date')['sales'].sum()
c1, c2 = st.columns(2)

wd = total_day.groupby(total_day.index.day_of_week).mean()
wd = (wd / wd.mean() - 1) * 100
fig = px.bar(x=WEEKDAYS, y=wd.values, color=wd.values > 0, color_discrete_map={True: 'seagreen', False: 'indianred'},
             labels={'x': '', 'y': '% เทียบค่าเฉลี่ย'})
fig.update_layout(title='วันในสัปดาห์: เสาร์–อาทิตย์ขายดีกว่า', showlegend=False, height=340, margin=dict(t=40, b=10))
c1.plotly_chart(fig, width='stretch')

foods = daily[(daily['cat_id'] == 'FOODS') & daily['state_id'].isin(states)]
foods = foods.groupby(['state_id', 'date', 'snap'])['sales'].sum().reset_index()
snap = foods.groupby(['state_id', 'snap'])['sales'].mean().unstack()
snap_lift = ((snap[1] / snap[0] - 1) * 100).reset_index(name='lift')
fig = px.bar(snap_lift, x='state_id', y='lift', text=snap_lift['lift'].map('{:+.0f}%'.format),
             labels={'state_id': 'รัฐ', 'lift': '% ยอดขายเพิ่ม'})
fig.update_traces(marker_color='seagreen')
fig.update_layout(title='วัน SNAP (คูปองอาหารรัฐ): ยอดขาย FOODS เพิ่มขึ้น', height=340, margin=dict(t=40, b=10))
c2.plotly_chart(fig, width='stretch')

# ---------- 3. โมเดลพยากรณ์ ----------
st.header('3. โมเดลพยากรณ์แม่นแค่ไหน')
c1, c2 = st.columns(2)
if models is not None:
    m = models.sort_values('WRMSSE', ascending=False)
    colors = ['seagreen' if x == models.iloc[0]['method'] else
              'gray' if x in ('Naive', 'Seasonal naive', 'Mean 28d') else 'steelblue' for x in m['method']]
    fig = go.Figure(go.Bar(x=m['WRMSSE'], y=m['method'], orientation='h', marker_color=colors,
                           text=m['WRMSSE'].map('{:.2f}'.format)))
    fig.update_layout(title='ความคลาดเคลื่อน (WRMSSE) ยิ่งสั้นยิ่งดี | เทา = วิธีง่ายแบบเดิม',
                      height=380, margin=dict(t=40, b=10))
    c1.plotly_chart(fig, width='stretch')
else:
    c1.info('ยังไม่มีผล: รัน notebook 02 ก่อน')

if FORECAST_PATH.exists():
    fc = pd.read_parquet(FORECAST_PATH)
    fc = fc[fc['store_id'].str[:2].isin(states) & fc['item_id'].str.rsplit('_', n=2).str[0].isin(cats)]
    fc_total = fc.drop(columns=['item_id', 'store_id']).sum()
    fc_total.index = pd.to_datetime(fc_total.index)
    hist = total_day.iloc[-90:]
    fig = go.Figure([go.Scatter(x=hist.index, y=hist.values, name='ยอดขายจริง', line_color='black'),
                     go.Scatter(x=fc_total.index, y=fc_total.values, name='พยากรณ์', line=dict(color='seagreen', width=3))])
    fig.update_layout(title='ยอดขาย (ชิ้น) 90 วันล่าสุด + พยากรณ์ 28 วันข้างหน้า', height=380, margin=dict(t=40, b=10))
    c2.plotly_chart(fig, width='stretch')
else:
    c2.info('ยังไม่มีไฟล์พยากรณ์ 28 วัน: รัน notebook 02 (ส่วน 2.7) ก่อน')

# ---------- 4. คุ้มค่าแค่ไหน ----------
st.header('4. คุ้มค่าแค่ไหน (คิดเป็นเงิน)')
if impact is not None:
    st.dataframe(impact.style.format('${:,.0f}'), width='stretch')
    st.caption('สมมติฐาน: กำไร 25%, ค่าถือครองสต็อก 25%/ปี, ของเสีย/ลดราคา 5% ของส่วนเกิน (รายละเอียดใน notebook 03)')
else:
    st.info('ยังไม่มีผล: รัน notebook 03 เพื่อคำนวณเงินที่ประหยัดได้')

# ---------- 5. ข้อเสนอ ----------
st.header('5. ข้อเสนอสำหรับผู้บริหาร')
st.markdown('''
| ทำอะไร | ทำไม |
|---|---|
| สั่งของตามพยากรณ์ของโมเดลแทนวิธีเดิม | ความคลาดเคลื่อนลดลง ลดทั้งของขาดและของล้น |
| เติมสต็อก FOODS ก่อนวัน SNAP โดยเฉพาะ WI | ยอดขายอาหารพุ่งขึ้นในวัน SNAP |
| เพิ่มพนักงาน/สต็อกวันเสาร์–อาทิตย์ | ยอดขายสูงกว่าวันธรรมดา ~20% |
| เริ่ม pilot: FOODS ในรัฐ CA 8–12 สัปดาห์ (A/B test) | พิสูจน์ผลจริงก่อนขยายทุกสาขา |
| วัดผลด้วย KPI: Out-of-stock rate, ของเสีย, inventory turnover | ติดตามว่าได้เงินคืนจริงตามที่คาด |
''')
