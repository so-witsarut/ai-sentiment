# ai-sentiment

Worker วิเคราะห์ sentiment และ intent ของโพสต์จาก Blue Eye REST API ทำงานต่อเนื่องบนเครื่องที่เปิดไว้ตลอดเวลา ดึงรายการที่รอวิเคราะห์ ส่งข้อความให้โมเดล แล้วส่งผลกลับผ่าน REST API โปรแกรมนี้ไม่เชื่อม MySQL หรือ MongoDB โดยตรง

ใช้ `ai_sentiment.py` รันต่อเนื่องสำหรับวันปัจจุบัน หรือระบุ `--from-date` และ `--to-date` เพื่อวิเคราะห์ช่วงวันที่ที่ต้องการหนึ่งรอบ

## สถาปัตยกรรม

```text
ai_sentiment.py / run.bat
  └─ วนรอบทุก RUN_INTERVAL_SECONDS (วันปัจจุบันตามเวลาไทย UTC+07:00)
       └─ GET /internal/sentiment/posts?date_from=...&date_to=...&page_size=500&page=1
            └─ ดึงหน้าถัดไปจนหมด
            └─ เลือก provider ตามเวลาไทยและสวิตช์ ENABLE_*
                 └─ วิเคราะห์ sentiment + intent; ลองโมเดลสำรองเมื่อจำเป็น
                      └─ POST /internal/sentiment/results เมื่อ SAVE_DB=true
```

โค้ดหลักอยู่ใน `ai_sentiment.py` ส่วน `run.bat` เป็นตัวช่วยเปิด worker บน Windows ค่าใช้งานอยู่ใน `.env`; `.env.example` เป็นแม่แบบสำหรับเครื่องใหม่

คำขอดึงโพสต์ใช้ `page_size=500` และเริ่มที่ `page=1` จากนั้นดึงหน้าถัดไปจนหมดก่อนเริ่มวิเคราะห์ หาก API ส่ง `sentiment_status` มาด้วย worker จะข้ามรายการสถานะ `1` และ `2` เพื่อไม่วิเคราะห์รายการที่มีสถานะแล้ว

| ส่วน | หน้าที่ |
|---|---|
| REST API | ดึงโพสต์ที่รอวิเคราะห์และรับผลลัพธ์กลับ |
| OpenRouter | เรียก Jev/DeepSeek และโมเดล `openrouter:` ที่ตั้งไว้ |
| Gemini | เรียกโมเดลที่ขึ้นต้นด้วย `api:` |
| Ollama | เรียกโมเดลที่ไม่มี prefix รวมถึง Ollama Cloud หากตั้งชื่อโมเดลนั้น |
| Worker | แบ่ง batch, จำกัดคำขอพร้อมกัน และวนรอบตามช่วงเวลาที่กำหนด |

เมื่อโพสต์มี `project_name` โปรแกรมวิเคราะห์ความรู้สึกและ intent **ต่อโปรเจกต์นั้น** คีย์เวิร์ดช่วยเลือกข้อความบางส่วนสำหรับส่งให้โมเดล แต่การมีคีย์เวิร์ดอย่างเดียวไม่ได้แปลว่าโพสต์กล่าวถึงโปรเจกต์ หากไม่มี `project_name` จะวิเคราะห์ภาพรวมของโพสต์แทน ข้อมูลแหล่งเผยแพร่และ URL ใช้เป็นบริบท; worker ไม่เปิดหน้าเว็บตาม URL นั้น Intent ที่รองรับคือ `complaint`, `information`, `recommendation` และ `enquiry`; หากผลไม่มี intent ที่ถูกต้อง จะใช้ `information`

## ติดตั้ง

ต้องมี Python และติดตั้ง dependency จาก `requirements.txt` ก่อนรัน คัดลอก `.env.example` เป็น `.env` แล้วกรอก token และ API key ของ provider ที่เปิดใช้งาน ห้ามนำ `.env` ขึ้น Git

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux/macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

หากใช้ `.env` ที่มีอยู่บนเครื่องนี้แล้ว ไม่ต้องคัดลอกแม่แบบทับไฟล์เดิม ค่า MySQL, MongoDB, SSH และ DigitalOcean เดิมถูกเก็บไว้ในหมวด Legacy ท้าย `.env` เพื่อให้สคริปต์อื่นที่อาจใช้ไฟล์เดียวกันยังอ่านได้; worker นี้ไม่ได้ใช้ค่าเหล่านั้น

## ตั้งค่า `.env`

ไฟล์จัดเป็นหมวด REST และการเชื่อมต่อ, สวิตช์ provider, โมเดล, ตารางเวลา, การวิเคราะห์, concurrency และการรัน ค่าที่ไม่มีใน `.env` จะใช้ค่าเริ่มต้นในโค้ด การแก้ค่าใน `.env` มีผลหลังรีสตาร์ต worker

| ค่า | ความหมาย |
|---|---|
| `BE_API_BASE_URL`, `BE_API_TOKEN` | ที่อยู่และ token ของ Blue Eye REST API; URL มีค่าเริ่มต้นในโค้ด |
| `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | API key ของ provider ที่เปิดใช้งาน |
| `OLLAMA_HOST`, `OLLAMA_MODEL` | ที่อยู่ Ollama และโมเดลเริ่มต้น; `OLLAMA_HOST` เริ่มต้นที่ `http://localhost:11434` |
| `ENABLE_OPENROUTER`, `ENABLE_GEMINI`, `ENABLE_OLLAMA` | `false` เพื่อข้าม provider นั้นทุกช่วงเวลา |
| `VALIDATION_MODELS` | รายชื่อโมเดลสำรองคั่นด้วยจุลภาค: ชื่อธรรมดา = Ollama, `api:` = Gemini, `openrouter:` = OpenRouter |
| `OPENROUTER_PROVIDERS`, `OPENROUTER_ALLOW_FALLBACKS` | เลือก endpoint ของ OpenRouter ตามลำดับ และกำหนดว่าจะให้ OpenRouter ใช้ endpoint อื่นต่อได้หรือไม่ |
| `PROVIDER_SCHEDULE_ENABLED`, `OPENROUTER_SCHEDULE_START`, `OPENROUTER_SCHEDULE_END` | เปิดตารางเวลาและกำหนดช่วง OpenRouter ตามเวลาไทย UTC+07:00 |
| `AI_COST_MODE`, `ENABLE_JEV_HYBRID`, `ENABLE_PROBABILISTIC_MODE`, `BYPASS_LOCAL_TRIAGE` | กำหนดเส้นทางและวิธีวิเคราะห์; ดูค่าใช้งานใน `.env.example` |
| `CONCURRENT_WORKERS`, `MAX_IN_FLIGHT`, `BATCH_SIZE` | จำนวน worker, งานที่ค้างพร้อมกัน และขนาด batch ต่อหนึ่ง process |
| `GEMINI_MAX_CONCURRENCY`, `OLLAMA_MAX_CONCURRENCY`, `DEEPSEEK_MAX_CONCURRENCY` | เพดานคำขอพร้อมกันของแต่ละ provider ต่อหนึ่ง process |
| `RUN_INTERVAL_SECONDS` | เวลารอระหว่างรอบ; หากไม่ระบุ ค่าเริ่มต้นคือ 5 วินาที |
| `SAVE_DB` | `true` จึงส่งผลไป REST API; `false` วิเคราะห์และแสดงผลโดยไม่ส่งผล |

เมื่อเปิดตารางเวลา ค่าในแม่แบบกำหนดให้ช่วง **08:00–21:59** ลอง OpenRouter ก่อน แล้วจึง Gemini และ Ollama; ช่วง **22:00–07:59** ลอง Gemini, Ollama แล้วจึง OpenRouter เป็นทางสำรอง ลำดับภายในแต่ละ provider อิง `VALIDATION_MODELS` และ provider ที่ปิดสวิตช์จะถูกข้าม เลือกลำดับหนึ่งครั้งเมื่อเริ่มโพสต์นั้น แม้เวลาจะข้ามช่วงระหว่างวิเคราะห์ก็ใช้ลำดับเดิมจนเสร็จ

`AI_COST_MODE=low` ให้ยอมรับผล Jev ได้มากขึ้นในบางกรณีเพื่อลดการเรียก DeepSeek; `standard` ใช้เกณฑ์ที่เข้มกว่า ค่าจำกัดความยาวข้อความและจำนวนครั้งที่ลองใหม่ของแต่ละ provider อยู่ใน `.env.example` การจำกัด concurrency มีผลต่อ **หนึ่ง process** เท่านั้น

## วิธีรัน

เริ่ม worker **เพียงหนึ่ง process** เพื่อไม่ให้จำนวนคำขอพร้อมกันคูณขึ้นและเสี่ยงประมวลผลโพสต์ซ้ำ หยุดด้วย `Ctrl+C` แล้วรอให้โปรแกรมออกก่อนเริ่มใหม่

| ระบบ | คำสั่ง | รายละเอียด |
|---|---|---|
| Windows PowerShell | `py ai_sentiment.py` | ใช้ Python ที่ `py` เลือก; ต้องติดตั้ง dependency ใน interpreter นั้น |
| Windows PowerShell | `.\.venv\Scripts\python.exe ai_sentiment.py` | ใช้ virtual environment ของโปรเจกต์ |
| Windows PowerShell | `.\.venv\Scripts\python.exe ai_sentiment.py --mode rest` | ระบุ REST mode ชัดเจน; ให้ผลเหมือนคำสั่งก่อนหน้า |
| Windows PowerShell | `.\run.bat` | ใช้ `.venv` และ `.env` ในโฟลเดอร์โปรเจกต์ |
| Windows PowerShell | `.\.venv\Scripts\python.exe ai_sentiment.py --from-date 2026-09-28 --to-date 2026-09-29` | วิเคราะห์ช่วงวันที่นี้หนึ่งรอบแล้วจบ |
| Linux/macOS | `.venv/bin/python ai_sentiment.py` | ใช้ virtual environment ของโปรเจกต์ |
| Linux/macOS | `.venv/bin/python ai_sentiment.py --from-date 2026-09-28 --to-date 2026-09-29` | วิเคราะห์ช่วงวันที่นี้หนึ่งรอบแล้วจบ |

`--mode rest` เป็นโหมดเดียวที่รองรับและเป็นค่าเริ่มต้น หากไม่ระบุวันที่ worker จะรันต่อเนื่อง โดยส่ง `date_from` กับ `date_to` เป็น **วันปัจจุบันวันเดียวตามเวลาไทย UTC+07:00** ในแต่ละรอบ หากระบุวันที่ต้องใส่ทั้ง `--from-date` และ `--to-date` ในรูปแบบ `YYYY-MM-DD` และวันเริ่มต้องไม่หลังวันสิ้นสุด คำสั่งแบบระบุวันที่ใช้ค่า `SAVE_DB` จาก `.env` เช่นเดียวกับ worker ปกติ

เริ่มตรวจการทำงานด้วย `SAVE_DB=false` ก่อน ค่า `false` **ยังดึงข้อมูลและเรียกโมเดลจริง** จึงอาจใช้เครดิตหรือมีค่าใช้จ่าย เพียงแต่ไม่ส่งผลไป `/internal/sentiment/results` เมื่อพร้อมบันทึกผล ให้ตั้ง `SAVE_DB=true` ใน `.env` และรีสตาร์ต worker

## ผลลัพธ์เมื่อวิเคราะห์ไม่สำเร็จ

Worker ปัจจุบันไม่อ่านหรือเขียน analysis cache ในการรันจริง หากทุก provider ล้มเหลวหรือไม่มีผลที่ใช้ได้ โปรแกรมส่งค่าเริ่มต้น `sentiment=neutral`, `intent=information`, `sentiment_score=0`, `sentiment_scores={"positive":0,"negative":0,"neutral":100,"model":"rule:provider_failure"}` และ `sentiment_status="1"` เมื่อ `SAVE_DB=true` รายการที่ API ส่งกลับมาพร้อมสถานะ `1` หรือ `2` จะถูกข้ามในรอบถัดไป

## พัฒนาและตรวจสอบ

ชุดทดสอบใช้ mock สำหรับ provider และ REST API ไม่ควรเรียก API จริง:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/probabilistic -p "test_*.py"
```

บน Linux/macOS เปลี่ยน interpreter เป็น `.venv/bin/python` ก่อนรันคำสั่งเดียวกัน
