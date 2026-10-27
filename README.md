# Walmart Sales Forecasting (M5)

อ้างอิง: https://www.kaggle.com/competitions/m5-forecasting-accuracy

## โปรเจกต์นี้ทำอะไร

พยากรณ์ยอดขาย**รายวันล่วงหน้า 28 วัน** ของสินค้า 3,049 รายการใน 10 สาขาของ Walmart (CA, TX, WI) รวม 30,490 รายการสินค้า×สาขา

**ทำไมสำคัญ:** ถ้าพยากรณ์ต่ำไปของจะขาดชั้นวางและเสียยอดขาย ถ้าพยากรณ์สูงไปสต็อกจะล้น เงินจม ต้องลดราคาระบาย และอาหารเน่าเสีย
การพยากรณ์ที่แม่นขึ้นช่วยเพิ่มรายได้ ลดต้นทุน และช่วยวางแผนพนักงานและโปรโมชัน

**วัดผลอย่างไร:** ใช้ WRMSSE ซึ่งวัดความแม่นทุกระดับ ตั้งแต่ยอดรวมทั้งบริษัทจนถึงสินค้า×สาขา และถ่วงน้ำหนักด้วยยอดขายเป็นเงิน

## การติดตั้ง Environment

ต้องใช้ Python 3.11 และ RAM 16 GB ขึ้นไป (ข้อมูลแบบ long มีประมาณ 59 ล้านแถว ต้องลดขนาด dtype ก่อนใช้)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name m5-forecasting --display-name "Python (m5-forecasting)"
```

ใน VS Code ให้เลือก kernel **Python (m5-forecasting)** เมื่อเปิด notebook
ไฟล์ CSV ต้นฉบับดาวน์โหลดได้จากหน้า Kaggle ด้านบน แล้ววางไว้ใน `data/raw/`
รันเทสต์ด้วยคำสั่ง `pytest`

## โครงสร้างโฟลเดอร์

```
data/
  raw/            ข้อมูลต้นฉบับจาก Kaggle (ห้ามแก้ไข)
  processed/      ข้อมูล long + ปฏิทิน + ราคา + feature (parquet สร้างใหม่ได้จาก src/)
  submission/     (ยังไม่ใช้: โปรเจกต์นี้ไม่ส่งผลเข้า Kaggle)
notebooks/        notebook สำรวจข้อมูลและทดลองโมเดล ตั้งชื่อขึ้นต้นด้วยเลขลำดับ
src/              โค้ดที่ใช้ซ้ำ: เตรียมข้อมูล, สร้าง feature, เทรนโมเดล, คำนวณ WRMSSE
tests/            เทสต์ของโค้ดใน src/ (โดยเฉพาะ WRMSSE ต้องตรงกับ Kaggle)
analysis/         ผลเปรียบเทียบโมเดล: คะแนน backtest, WRMSSE แยกระดับ/หมวด, feature importance
reports/          ผลงานสำหรับธุรกิจ: insight, business impact, ข้อเสนอแนะ
  figures/        กราฟที่ใช้ใน report
MODEL_SELECTION_METHOD.md   วิธีพยากรณ์ วัดผล และเกณฑ์เลือกโมเดล
```

| โฟลเดอร์ | ใครสร้าง | เก็บใน git |
|---|---|---|
| `data/` | ดาวน์โหลด / โค้ดสร้าง | ไม่เก็บ (ไฟล์ใหญ่) |
| `notebooks/`, `src/`, `tests/` | คนเขียน | เก็บ |
| `analysis/`, `reports/` | โค้ดสร้าง + คนสรุป | เก็บ |

### data/raw: ข้อมูลต้นฉบับ

| ไฟล์ | เนื้อหา |
|---|---|
| `sales_train_validation.csv` | ยอดขายรายวันของสินค้าแต่ละรายการในแต่ละสาขา d_1–d_1913 (30,490 แถว) |
| `sales_train_evaluation.csv` | เหมือนไฟล์ด้านบน แต่มีข้อมูลถึง d_1941 (ใช้ train ตัวจริง) |
| `calendar.csv` | วันที่ 2011-01-29 ถึง 2016-06-19, วันในสัปดาห์, เทศกาล/event, วัน SNAP แต่ละรัฐ |
| `sell_prices.csv` | ราคาขายรายสัปดาห์ของสินค้าแต่ละรายการในแต่ละสาขา |
| `sample_submission.csv` | ตัวอย่างรูปแบบไฟล์ส่ง (คอลัมน์ id, F1–F28) |

## ขั้นตอนการทำงาน

แต่ละขั้นมี notebook ใน `notebooks/` ให้รันตามลำดับ เพราะ notebook ถัดไปใช้ผลลัพธ์ของ notebook ก่อนหน้า
ครั้งแรกที่รันจะสร้าง `data/processed/m5_long.parquet` จากข้อมูลดิบให้อัตโนมัติ (`src/data.py`)

| ขั้น | Notebook | ผลลัพธ์ |
|---|---|---|
| 1 | [01_sales_insights.ipynb](notebooks/01_sales_insights.ipynb) | กราฟใน `reports/figures/` |
| 2 | [02_forecast.ipynb](notebooks/02_forecast.ipynb) | ตารางเปรียบเทียบใน `analysis/`, ผลพยากรณ์ 28 วันใน `data/processed/` |
| 3 | [03_business_impact.ipynb](notebooks/03_business_impact.ipynb) | `analysis/business_impact.csv` |
| 4 | [04_recommendations.ipynb](notebooks/04_recommendations.ipynb) | สรุปข้อเสนอ |

### 1. Sales insights: อะไรขับเคลื่อนยอดขาย

- ยอดขายแยกตามรัฐ / สาขา / หมวด (กลุ่มไหนทำเงินมากที่สุด)
- ผลของวัน SNAP ต่อยอดขายอาหาร
- ผลของเทศกาลและวันในสัปดาห์
- ผลของราคาต่อยอดขาย
- สัดส่วนสินค้าขายเร็วกับสินค้าขายช้า

### 2. Forecast: พยากรณ์ 28 วัน

เปรียบเทียบ Baseline, LightGBM, XGBoost, CatBoost และ Random Forest ด้วย WRMSSE จาก backtest 3 รอบ
รายละเอียดวิธีการและเกณฑ์การเลือกอยู่ใน [MODEL_SELECTION_METHOD.md](MODEL_SELECTION_METHOD.md)

### 3. Business impact: ผลกระทบคิดเป็นเงิน

จำลองการสั่งของตามพยากรณ์ แล้วเทียบโมเดลที่เลือกกับ baseline:
- ยอดขายที่เสียไปเพราะของไม่พอ (lost sales)
- สต็อกส่วนเกินและต้นทุนการถือครอง (excess inventory)

> dataset ไม่มีข้อมูลต้นทุนและสต็อกจริง จึงต้องระบุสมมติฐานทุกข้อให้ชัดเจน (เช่น carrying cost 25% ต่อปี)

### 4. Recommendations: ข้อเสนอสำหรับผู้บริหาร

- Action ที่ได้จาก insight (เช่น เตรียมสต็อกก่อนวัน SNAP, เพิ่มพนักงานวันหยุด)
- แผน pilot (FOODS ในรัฐ CA) + A/B test
- KPI: Out-of-Stock rate, waste, inventory turnover, WRMSSE
- Roadmap: Pilot → Scale → MLOps

## License

โค้ดใช้ MIT License (ดู [LICENSE](LICENSE)) ส่วนข้อมูล M5 อยู่ภายใต้กติกาการแข่งขันของ Kaggle
