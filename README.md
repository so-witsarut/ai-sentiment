# ai-sentiment

Worker วิเคราะห์ sentiment และ intent ของโพสต์จาก Blue Eye REST API ทำงานต่อเนื่องบนเครื่องที่เปิดไว้ตลอดเวลา ดึงรายการที่รอวิเคราะห์ ส่งข้อความให้โมเดล แล้วส่งผลกลับผ่าน REST API โปรแกรมนี้ไม่เชื่อม MySQL หรือ MongoDB โดยตรง

ใช้ `ai_sentiment.py` รันต่อเนื่องสำหรับวันปัจจุบัน หรือระบุ `--from-date` และ `--to-date` เพื่อวิเคราะห์ช่วงวันที่ที่ต้องการหนึ่งรอบ


## สถาปัตยกรรม

```text
ai_sentiment.py / run.bat
  └─ วนรอบทุก RUN_INTERVAL_SECONDS (วันปัจจุบันตามเวลาไทย UTC+07:00)
       └─ GET /internal/sentiment/posts?date_from=...&date_to=...&page_size=500&page=1
            └─ วิเคราะห์และบันทึกทีละหน้า แล้วจึงดึงข้อมูลต่อ
            └─ เลือก provider ตามเวลาไทยและสวิตช์ ENABLE_*
                 └─ วิเคราะห์ sentiment + intent; ลองโมเดลสำรองเมื่อจำเป็น
                      └─ POST /internal/sentiment/results เมื่อ SAVE_DB=true
```

โค้ดหลักอยู่ใน `ai_sentiment.py` ส่วน `run.bat` เป็นตัวช่วยเปิด worker บน Windows ค่าใช้งานอยู่ใน `.env`; `.env.example` เป็นแม่แบบสำหรับเครื่องใหม่

คำขอดึงโพสต์ใช้ `page_size=500` และเริ่มที่ `page=1` เมื่อได้แต่ละหน้าจะเริ่มวิเคราะห์และบันทึกเป็น batch ทันที ไม่รอดึงข้อมูลทั้งช่วงวันที่ให้ครบก่อน หาก API ส่ง `sentiment_status` มาด้วย worker จะข้ามรายการสถานะ `1` และ `2` เมื่อบันทึกสำเร็จจะตรวจหน้าเดิมอีกครั้งเพื่อรับรายการที่เลื่อนเข้ามาใน API ที่แสดงเฉพาะรายการรอวิเคราะห์ โดยจำ ID ที่เคยประมวลผลในรอบนี้เพื่อไม่ส่ง AI ซ้ำ หากไม่มีรายการใหม่จึงไปหน้าถัดไป ส่วน dry run จะดึงหน้าถัดไปตามปกติ หากดึงหน้าหลัง ๆ ไม่สำเร็จ ผลที่บันทึกไปแล้วจะยังคงอยู่

| ส่วน | หน้าที่ |
|---|---|
| REST API | ดึงโพสต์ที่รอวิเคราะห์และรับผลลัพธ์กลับ |
| OpenRouter | เรียก Jev/DeepSeek และโมเดล `openrouter:` ที่ตั้งไว้ |
| Gemini | เรียกโมเดลที่ขึ้นต้นด้วย `api:` |
| Ollama | เรียกโมเดลที่ไม่มี prefix รวมถึง Ollama Cloud หากตั้งชื่อโมเดลนั้น |
| Worker | แบ่ง batch, จำกัดคำขอพร้อมกัน และวนรอบตามช่วงเวลาที่กำหนด |

เมื่อโพสต์มี `project_name` โปรแกรมวิเคราะห์ความรู้สึก **ต่อโปรเจกต์นั้น** ส่วน intent คือ **จุดประสงค์หลักของทั้งโพสต์** แม้ข้อความไม่เกี่ยวกับโปรเจกต์ เช่น โพสต์ร้องเรียนร้านอาหารที่ไม่เกี่ยวกับ BLCP จะได้ sentiment ต่อ BLCP เป็น neutral และ intent เป็น complaint หากไม่มี `project_name` จะวิเคราะห์ sentiment ภาพรวมของโพสต์แทน ข้อมูลแหล่งเผยแพร่และ URL ใช้เป็นบริบท; worker ไม่เปิดหน้าเว็บตาม URL นั้น Intent ที่รองรับคือ `complaint`, `information`, `recommendation` และ `enquiry`; pure praise เป็น information และคำถามเชิงร้องเรียนเป็น complaint หากผลไม่มี intent ที่ถูกต้อง จะใช้ `information`

คีย์เวิร์ดรับได้หลายคำทั้งรายการ เช่น `["ค่าไฟ", "โรงไฟฟ้า"]` และข้อความคั่น comma เช่น `"ค่าไฟ, โรงไฟฟ้า"` โปรแกรมตัดค่าว่างและคำซ้ำ ใช้ `full_text` ก่อน `content` เมื่อ API ส่งมาทั้งคู่ แล้วเลือกบริบทรอบ Target และหลายคีย์เวิร์ดที่พบจริง รวมช่วงที่ซ้อนทับและรักษาลำดับข้อความเดิมภายใต้ `JEV_TEXT_MAX_CHARS` และ `DEEPSEEK_TEXT_MAX_CHARS` ข้อความสั้นสำหรับแสดง log แยกจากข้อความต้นฉบับ คีย์เวิร์ดช่วยเลือก excerpt แต่ไม่ใช่หลักฐานว่าเกี่ยวข้องกับ Target หาก API ส่งมาเฉพาะ excerpt จะใช้ได้เท่าที่ได้รับ

ความกำกวมไม่ถูกแปลงเป็น negative อัตโนมัติ: คะแนน `AMBIGUOUS_OR_IRONY` รวมอยู่ใน neutral ขณะที่ raw probabilities และ `irony_score` ยังเก็บไว้ภายใน หากกลุ่มนี้เป็นคะแนนสูงสุด จะมี `is_ambiguous=true` และ reason อธิบายความกำกวม; `irony_score` จึงไม่ใช่หลักฐานว่าพบการประชด โมเดลได้รับคำสั่งให้ตีความประชดที่เจตนาชัดเป็น positive/negative ตามความหมาย และกระจายคะแนนบวก/ลบสำหรับ mixed กรณี Jev ให้ความกำกวมเป็นคำตอบหลักในโพสต์ที่เกี่ยวข้อง จะส่งต่อโมเดลสำรองแม้ใช้โหมด low ส่วน REST payload ใช้ field เดิม

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
| `OLLAMA_DEEP_TIMEOUT`, `OLLAMA_THINK`, `OLLAMA_KEEP_ALIVE` | timeout ต่อคำขอของ Ollama (เริ่มต้น 30 วินาที), คุมโหมด thinking (`auto` ปิดให้ DeepSeek-R1 อัตโนมัติ) และเวลาค้างโมเดลในหน่วยความจำ |
| `ENABLE_OPENROUTER`, `ENABLE_GEMINI`, `ENABLE_OLLAMA` | `false` เพื่อข้าม provider นั้นทุกช่วงเวลา |
| `VALIDATION_MODELS` | รายชื่อโมเดลสำรองคั่นด้วยจุลภาค: ชื่อธรรมดา = Ollama, `api:` = Gemini, `openrouter:` = OpenRouter |
| `OPENROUTER_PROVIDERS`, `OPENROUTER_ALLOW_FALLBACKS` | เลือก endpoint ของ OpenRouter ตามลำดับ และกำหนดว่าจะให้ OpenRouter ใช้ endpoint อื่นต่อได้หรือไม่ |
| `PROVIDER_SCHEDULE_ENABLED`, `OPENROUTER_SCHEDULE_START`, `OPENROUTER_SCHEDULE_END` | เปิดตารางเวลาและกำหนดช่วง OpenRouter ตามเวลาไทย UTC+07:00 |
| `AI_COST_MODE`, `ENABLE_JEV_HYBRID`, `ENABLE_PROBABILISTIC_MODE`, `BYPASS_LOCAL_TRIAGE` | กำหนดเส้นทางและวิธีวิเคราะห์; ดูค่าใช้งานใน `.env.example` |
| `CONCURRENT_WORKERS`, `MAX_IN_FLIGHT`, `BATCH_SIZE` | จำนวน worker, งานที่ค้างพร้อมกัน และขนาด batch ต่อหนึ่ง process |
| `GEMINI_MAX_CONCURRENCY`, `OLLAMA_MAX_CONCURRENCY`, `DEEPSEEK_MAX_CONCURRENCY` | เพดานคำขอพร้อมกันของแต่ละ provider ต่อหนึ่ง process |
| `GEMINI_RATE_LIMIT_ENABLED`, `GEMINI_RATE_LIMIT_PERCENT` | เปิดการคุม RPM/TPM ของ Gemini; ค่าเริ่มต้น `true` และใช้เพดาน 90% เพื่อเผื่อโควตา |
| `GEMINI_RATE_LIMITS` | JSON ระบุ `rpm` และ `tpm` แยกตามชื่อโมเดล API; ตัวอย่างอยู่ใน `.env.example` |
| `GEMINI_DEFAULT_RPM`, `GEMINI_DEFAULT_TPM` | เพดานโมเดลที่ไม่ได้ระบุในรายการ เริ่มต้น 5 RPM / 16,000 TPM ก่อนเผื่อ 10% |
| `RUN_INTERVAL_SECONDS` | เวลารอระหว่างรอบ; หากไม่ระบุ ค่าเริ่มต้นคือ 5 วินาที |
| `SAVE_DB` | `true` จึงส่งผลไป REST API; `false` วิเคราะห์และแสดงผลโดยไม่ส่งผล |

เมื่อเปิดตารางเวลา ค่าในแม่แบบกำหนดให้ช่วง **10:00–21:59** ลอง Jev ผ่าน OpenRouter โดยมีเพดาน **2 คำขอ Jev รวมต่อโพสต์** และไม่ลอง Jev fallback aliases ซ้อนอีก หากคำขอแรก timeout, ได้ error ที่ retry ได้ หรือผลผิดรูปแบบ จะลอง Jev ซ้ำได้อีกครั้ง; error ถาวรอย่าง 401/403/400/404 จะส่งต่อทันที เมื่อ Jev ตอบสำเร็จแต่ความมั่นใจต่ำหรือมี conflict จะส่งต่อทันทีโดยไม่เรียก Jev ซ้ำ ขั้นส่งต่อใช้ DeepSeek ผ่าน OpenRouter แล้วจึงลองโมเดล Gemini และ Ollama ที่กำหนดไว้ตามลำดับ หยุดเมื่อได้ผลที่ใช้ได้

ช่วง **22:00–07:59** ข้าม Jev และลองโมเดล Gemini ทั้งกลุ่มก่อน Ollama แล้วจึง DeepSeek ผ่าน OpenRouter หากยังใช้ไม่ได้จึงลองโมเดล `openrouter:` ใน `VALIDATION_MODELS` เป็นทางสำรองต่อ ลำดับภายในแต่ละกลุ่มอิง `VALIDATION_MODELS` และ provider ที่ปิดสวิตช์จะถูกข้าม เลือกช่วงเวลาหนึ่งครั้งเมื่อเริ่มโพสต์นั้น แม้เวลาจะข้ามช่วงระหว่างวิเคราะห์ก็ใช้ลำดับเดิมจนเสร็จ

Log `[AI Usage]` แสดงจำนวนคำขอ OpenRouter, Gemini, Ollama และจำนวน `rule:provider_failure` แยกตาม profile กลางวัน/กลางคืน โดยนับคำขอที่ล้มเหลวด้วย รวมการเรียก Ollama ผ่าน generate แล้วตามด้วย chat ข้อมูลนี้อยู่ใน `telemetry.profiles` ของผล batch ด้วย

`AI_COST_MODE=low` ให้ยอมรับผล Jev ได้มากขึ้นในบางกรณีเพื่อลดการเรียก DeepSeek; `standard` ใช้เกณฑ์ที่เข้มกว่า ค่าจำกัดความยาวข้อความและจำนวนครั้งที่ลองใหม่ของแต่ละ provider อยู่ใน `.env.example` โดย `JEV_MAX_RETRIES` และ `JEV_FALLBACK_MODELS` ใช้เมื่อปิดตารางเวลา ส่วนช่วงกลางวันใช้เพดาน 2 คำขอข้างต้น การจำกัด concurrency มีผลต่อ **หนึ่ง process** เท่านั้น

### รูปแบบการตั้งค่า Provider ที่แนะนำ (Configuration Presets)

สามารถสลับการตั้งค่าใน `.env` ระหว่างการรันเฉพาะ **Gemini** (ฟรี/ประหยัดค่าใช้จ่าย) หรือรันคู่กับ **OpenRouter** (เสถียรสูงสุดและรองรับงานหนัก) ได้ดังนี้:

#### รูปแบบที่ 1: ใช้ GEMINI อย่างเดียว (Gemini-Only / ฟรี 100%)
เหมาะสำหรับช่วงที่ไม่มีเครดิต OpenRouter โดยใช้ Google Gemini Free Tier ล้วน:
- **`VALIDATION_MODELS`**: ให้ใช้โมเดล Flash ตัวเต็ม (`api:gemini-3.5-flash`, `api:gemini-3.6-flash`, `api:gemini-3.7-flash`) เนื่องจากโมเดลตระกูล `-lite` มีเพดาน Free Tier เพียง 500 requests/วัน
- **`CONCURRENT_WORKERS=4`**: ปรับลด worker เพื่อไม่ให้คำขอเข้ามาแย่งคิวจนเกินโควตา 15 RPM
- **`GEMINI_MAX_WAIT_SECONDS=60`**: ขยายเวลารอคิวเพื่อป้องกันไม่ให้คำขอตัดตกเป็น `rule:provider_failure`

```env
# 2. Provider switches
ENABLE_OPENROUTER=false
ENABLE_OLLAMA=false
ENABLE_GEMINI=true

# 3. Models and OpenRouter routing
VALIDATION_MODELS=api:gemini-3.5-flash,api:gemini-3.6-flash,api:gemini-3.7-flash,api:gemma-4-26b-a4b-it

# 4. Provider order by Thailand time (UTC+07:00)
PROVIDER_SCHEDULE_ENABLED=false

# 6. Concurrency per worker process
CONCURRENT_WORKERS=4
MAX_IN_FLIGHT=6
BATCH_SIZE=10
GEMINI_MAX_CONCURRENCY=2
GEMINI_RATE_LIMIT_ENABLED=true
GEMINI_RATE_LIMIT_PERCENT=90
GEMINI_MAX_WAIT_SECONDS=60
GEMINI_DEFAULT_RPM=15
GEMINI_DEFAULT_TPM=250000
GEMINI_RATE_LIMITS='{"gemini-3.5-flash":{"rpm":15,"tpm":250000},"gemini-3.6-flash":{"rpm":15,"tpm":250000},"gemini-3.7-flash":{"rpm":15,"tpm":250000},"gemma-4-26b-a4b-it":{"rpm":30,"tpm":16000}}'
OLLAMA_MAX_CONCURRENCY=1
```

#### รูปแบบที่ 2: เปิดทั้ง OPENROUTER และ GEMINI (Hybrid / เสถียรสูงสุด)
เหมาะสำหรับช่วงที่มีเครดิต OpenRouter โดยให้ระบบสลับการทำงานอัตโนมัติตามตารางเวลา หรือใช้ OpenRouter เป็นตาข่ายสำรองทันทีเมื่อ Gemini มีปัญหา:
- ช่วงกลางวัน (10:00–21:59): OpenRouter วิ่งเป็นด่านแรกด้วยความเร็วสูง
- ช่วงกลางคืน (22:00–09:59): Gemini วิ่งเป็นด่านแรกเพื่อประหยัดเครดิต
- รองรับงานปริมาณมากด้วย `CONCURRENT_WORKERS=8` และ `DEEPSEEK_MAX_CONCURRENCY=6`

```env
# 2. Provider switches
ENABLE_OPENROUTER=true
ENABLE_OLLAMA=false
ENABLE_GEMINI=true

# 3. Models and OpenRouter routing
VALIDATION_MODELS=api:gemini-3.5-flash,api:gemini-3.6-flash,openrouter:deepseek/deepseek-chat,api:gemma-4-26b-a4b-it

# 4. Provider order by Thailand time (UTC+07:00)
PROVIDER_SCHEDULE_ENABLED=true
OPENROUTER_SCHEDULE_START=10:00
OPENROUTER_SCHEDULE_END=22:00

# 6. Concurrency per worker process
CONCURRENT_WORKERS=8
MAX_IN_FLIGHT=12
BATCH_SIZE=15
DEEPSEEK_MAX_CONCURRENCY=6
GEMINI_MAX_CONCURRENCY=2
GEMINI_RATE_LIMIT_ENABLED=true
GEMINI_RATE_LIMIT_PERCENT=90
GEMINI_MAX_WAIT_SECONDS=60
GEMINI_DEFAULT_RPM=15
GEMINI_DEFAULT_TPM=250000
GEMINI_RATE_LIMITS='{"gemini-3.5-flash":{"rpm":15,"tpm":250000},"gemini-3.6-flash":{"rpm":15,"tpm":250000},"gemini-3.7-flash":{"rpm":15,"tpm":250000}}'
OLLAMA_MAX_CONCURRENCY=1
```

#### รูปแบบที่ 3: วิเคราะห์ย้อนหลัง / ใช้ Ollama ภายในเครื่อง (`deepseek-r1:8b`)
เหมาะสำหรับการวิเคราะห์ย้อนหลังปริมาณมากโดยไม่เสียค่า API หรือไม่ติดโควตา Free Tier:
- **`VALIDATION_MODELS=deepseek-r1:8b`** และ **`OLLAMA_MODEL=deepseek-r1:8b`**: ชี้ไปที่โมเดล DeepSeek-R1 บนเครื่องโลคอล
- **`OLLAMA_THINK=false`**: ปิดโหมด thinking ระหว่างสร้าง JSON เพื่อให้ตอบใน ~10–15 วินาทีต่อโพสต์แทนการรอ timeout (โปรแกรมปิดให้อัตโนมัติสำหรับ `deepseek-r1` หากไม่กำหนดค่า)
- **`OLLAMA_DEEP_TIMEOUT=90`** และ **`OLLAMA_MAX_CONCURRENCY=1`**: เผื่อเวลาสำหรับข้อความยาว 1,200 ตัวอักษร และล็อกให้ประมวลผลทีละ 1 คำขอเพื่อไม่ให้ RAM/VRAM เต็ม
- **`BATCH_SIZE=5`**: บันทึกผลเป็นระยะทุก ~1 นาที รันย้อนหลังได้ทันทีด้วย `.\run.bat --from-date YYYY-MM-DD --to-date YYYY-MM-DD`

```env
# 2. Provider switches
ENABLE_OPENROUTER=false
ENABLE_OLLAMA=true
ENABLE_GEMINI=false

# 3. Models and OpenRouter routing
VALIDATION_MODELS=deepseek-r1:8b
OLLAMA_MODEL=deepseek-r1:8b

# 4. Provider order by Thailand time (UTC+07:00)
PROVIDER_SCHEDULE_ENABLED=false

# 5. Analysis and provider requests
OLLAMA_DEEP_TIMEOUT=90
OLLAMA_THINK=false
OLLAMA_KEEP_ALIVE=15m

# 6. Concurrency per worker process
CONCURRENT_WORKERS=2
MAX_IN_FLIGHT=4
BATCH_SIZE=5
OLLAMA_MAX_CONCURRENCY=1
```

## วิธีรัน

Gemini จำกัด RPM และ TPM แยกตามโมเดลด้วยหน้าต่างย้อนหลัง 60 วินาที และแชร์ตัวนับระหว่างทุก thread/analyzer ใน process เดียวกัน เมื่อโควตาไม่พอจะรอก่อนส่ง โดยไม่ยึดช่อง HTTP ของโมเดลอื่น นับทุกคำขอรวม retry และ fallback หลัง HTTP 400; เมื่อเจอ 429 จะพักโมเดลนั้นตาม `Retry-After` แบบตัวเลขหรือ `RetryInfo.retryDelay` หากไม่มีใช้ 60 วินาที แม้ไม่มีรอบ retry เหลือ

TPM นับเฉพาะ input ตาม [เอกสาร Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) ก่อนส่งจองโทเคนแบบเผื่อจากจำนวน UTF-8 bytes ของ system instruction และข้อความพร้อม overhead แล้วปรับตาม `usageMetadata.promptTokenCount` เมื่อ API ส่งกลับมา จึงอาจรอมากกว่าที่จำเป็น โดยเฉพาะข้อความภาษาไทย หากประมาณการคำขอเดียวเกินงบทั้งนาทีจะข้ามโมเดลนั้นไปลองตัวถัดไปแทนการรอไม่สิ้นสุด คำขอที่ล้มเหลวหรือไม่มี usage ยังคงนับงบที่จองไว้ และเก็บงบไว้อีก 60 วินาทีหลังคำขอจบเพื่อเผื่อเวลาฝั่ง server

ค่าตัวอย่างอิงโควตาโปรเจกต์ใน AI Studio ที่ได้รับมา ปรับ `GEMINI_RATE_LIMITS` เมื่อโควตาเปลี่ยน ตัวคุมนี้ไม่รวมการใช้งานจาก process/แอปอื่นและไม่คุม RPD จึงยังอาจได้รับ 429 จากโควตารวมของโปรเจกต์ ตัวนับเริ่มใหม่เมื่อรีสตาร์ต; หากเพิ่งใช้โควตาจนเต็มให้เว้นอย่างน้อย 60 วินาทีก่อนเริ่ม worker ใหม่ การแก้โค้ดหรือ `.env` ต้องรีสตาร์ต worker จึงมีผล

เริ่ม worker **เพียงหนึ่ง process** เพื่อไม่ให้จำนวนคำขอพร้อมกันคูณขึ้นและเสี่ยงประมวลผลโพสต์ซ้ำ หยุดด้วย `Ctrl+C` แล้วรอให้โปรแกรมออกก่อนเริ่มใหม่

| ระบบ | คำสั่ง | รายละเอียด |
|---|---|---|
| Windows PowerShell | `py ai_sentiment.py` | ใช้ Python ที่ `py` เลือก; ต้องติดตั้ง dependency ใน interpreter นั้น |
| Windows PowerShell | `.\.venv\Scripts\python.exe ai_sentiment.py` | ใช้ virtual environment ของโปรเจกต์ |
| Windows PowerShell | `.\.venv\Scripts\python.exe ai_sentiment.py --mode rest` | ระบุ REST mode ชัดเจน; ให้ผลเหมือนคำสั่งก่อนหน้า |
| Windows PowerShell | `.\run.bat` | ใช้ `.venv` และ `.env` ในโฟลเดอร์โปรเจกต์ |
| Windows PowerShell | `.\run.bat --from-date 2026-09-28 --to-date 2026-09-29` | วิเคราะห์ช่วงวันที่นี้หนึ่งรอบผ่าน `run.bat` |
| Windows PowerShell | `.\.venv\Scripts\python.exe ai_sentiment.py --from-date 2026-09-28 --to-date 2026-09-29` | วิเคราะห์ช่วงวันที่นี้หนึ่งรอบแล้วจบ |
| Windows PowerShell | `.\run_sentiment_backfill.bat --from-date 2026-11-01 --to-date 2026-11-16 --save-db` | สคริปต์รันย้อนหลังเฉพาะ Ollama (`qwen3:8b`) ตามกติกา `sentiment_worker.py` โดยไม่บันทึก `rule:provider_failure` |
| Linux/macOS | `.venv/bin/python ai_sentiment.py` | ใช้ virtual environment ของโปรเจกต์ |
| Linux/macOS | `.venv/bin/python ai_sentiment.py --from-date 2026-09-28 --to-date 2026-09-29` | วิเคราะห์ช่วงวันที่นี้หนึ่งรอบแล้วจบ |

`--mode rest` เป็นโหมดเดียวที่รองรับและเป็นค่าเริ่มต้น หากไม่ระบุวันที่ worker จะรันต่อเนื่อง โดยส่ง `date_from` กับ `date_to` เป็น **วันปัจจุบันวันเดียวตามเวลาไทย UTC+07:00** ในแต่ละรอบ หากระบุวันที่ต้องใส่ทั้ง `--from-date` และ `--to-date` ในรูปแบบ `YYYY-MM-DD` และวันเริ่มต้องไม่หลังวันสิ้นสุด คำสั่งแบบระบุวันที่ใช้ค่า `SAVE_DB` จาก `.env` เช่นเดียวกับ worker ปกติ

เริ่มตรวจการทำงานด้วย `SAVE_DB=false` ก่อน ค่า `false` **ยังดึงข้อมูลและเรียกโมเดลจริง** จึงอาจใช้เครดิตหรือมีค่าใช้จ่าย เพียงแต่ไม่ส่งผลไป `/internal/sentiment/results` เมื่อพร้อมบันทึกผล ให้ตั้ง `SAVE_DB=true` ใน `.env` และรีสตาร์ต worker

## สคริปต์รันย้อนหลังเฉพาะ Ollama ตามกติกาหลัก (`sentiment_backfill.py`)

สำหรับรันวิเคราะห์ย้อนหลังบนเครื่องที่มีการ์ดจอแยก (เช่น RTX 5060 Ti 8GB) ด้วย `qwen3:8b` โดยเฉพาะ โดยอิงกติกา **Author's Own Voice** และโครงสร้างผลลัพธ์เดียวกับ `sentiment_worker.py` (`sentiment`, `sentiment_reason`, `sentiment_scores` พร้อม `multi_target`, `intent`, `intent_reason`, `incident`):

- ใช้เฉพาะ Ollama เท่านั้น (ไม่อ่าน `ENABLE_OPENROUTER`, `ENABLE_GEMINI`, `VALIDATION_MODELS`)
- **โหมดบันทึกจริง (`--save-db`)** ดึงคิวผ่าน `/internal/sentiment/pending?claim=true` (หรือใช้ `--missing incomplete` เพื่อดึงแถวที่บันทึกแล้วแต่ข้อมูลไม่ครบ/ติด `rule:` จาก `/internal/sentiment/incomplete` แล้วส่งทับผ่าน `/internal/sentiment/replace`) ส่วน `--dry-run` อ่านแบบ read-only โดยไม่เขียน DB
- **ไม่มีการสร้างหรือบันทึก `rule:provider_failure`**: หาก Ollama ค้าง, timeout หรือตอบไม่ผ่าน validation หลังลองซ้ำ 3 รอบ (`/api/generate` + `format="json"` โดยสลับ temperature/seed) จะข้ามโพสต์นั้นไว้ในสถานะเดิม และบันทึก ID/สาเหตุลง `reports/sentiment_backfill_failures.jsonl`
- มี Circuit Breaker หยุดทำงานอัตโนมัติ (ค่าเริ่มต้น 5 โพสต์ติดกัน) และบันทึก batch ที่สำเร็จก่อนหน้าเสมอ

```powershell
# ทดสอบ Dry-run ย้อนหลัง 7 วัน (จำกัดดู 5 โพสต์แรก)
.\.venv\Scripts\python.exe sentiment_backfill.py --days 7 --dry-run --limit 5

# รันจริงและบันทึกผลตามช่วงวันที่ (เฉพาะคิวใหม่ sentiment_status=0)
.\.venv\Scripts\python.exe sentiment_backfill.py --from-date 2026-11-01 --to-date 2026-11-16 --save-db

# รันซ่อม/เขียนทับรายการที่ status=1 แล้วแต่ข้อมูลยังไม่ครบหรือเป็น rule: (/incomplete -> /replace)
.\.venv\Scripts\python.exe sentiment_backfill.py --from-date 2026-11-01 --to-date 2026-11-16 --missing incomplete --save-db

# หรือใช้ตัวช่วยบน Windows (ส่งพารามิเตอร์หรือดับเบิลคลิกเพื่อกรอกวันที่)
.\run_sentiment_backfill.bat --from-date 2026-11-01 --to-date 2026-11-16 --save-db
```

## ผลลัพธ์เมื่อวิเคราะห์ไม่สำเร็จ

Worker ปัจจุบันไม่อ่านหรือเขียน analysis cache ในการรันจริง หากทุก provider ล้มเหลวหรือไม่มีผลที่ใช้ได้ โปรแกรมส่งค่าเริ่มต้น `sentiment=neutral`, `intent=information`, `sentiment_score=0`, `sentiment_scores={"positive":0,"negative":0,"neutral":100,"model":"rule:provider_failure"}` และ `sentiment_status="1"` เมื่อ `SAVE_DB=true` รายการที่ API ส่งกลับมาพร้อมสถานะ `1` หรือ `2` จะถูกข้ามในรอบถัดไป

## พัฒนาและตรวจสอบ

ชุดทดสอบใช้ mock สำหรับ provider และ REST API ไม่ควรเรียก API จริง:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/probabilistic -p "test_*.py"
```

บน Linux/macOS เปลี่ยน interpreter เป็น `.venv/bin/python` ก่อนรันคำสั่งเดียวกัน
