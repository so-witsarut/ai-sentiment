# Prompt ที่ส่งจริงใน REST flow ปัจจุบัน

เอกสารนี้เป็นภาพของโค้ดปัจจุบันใน `ai_sentiment.py` สำหรับโพสต์ที่ดึงจาก `/internal/sentiment/pending` แล้วส่งเข้า Jev และเมื่อจำเป็นจึงส่งต่อ DeepSeek ผ่าน OpenRouter ข้อความภาษาอังกฤษในส่วน **Jev questions** และ **DeepSeek system** ด้านล่างถอดตาม string ที่โค้ดส่งจริง ไม่มีการแปลหรือย่อ

## 1. ข้อมูลที่ใส่ใน prompt

REST flow ตั้ง `_analysis_scope="keyword"` ให้ทุกโพสต์ก่อนวิเคราะห์; เมื่อไม่มี Target ตัววิเคราะห์เปลี่ยนเป็น `overall` และใช้ prompt ในข้อ 4 ค่าที่ส่งมีดังนี้:

| ตัวแปร | ที่มาและพฤติกรรมปัจจุบัน |
|---|---|
| `Target` | ใช้ `project_name` เท่านั้น หากไม่มีจะวิเคราะห์ภาพรวมของโพสต์ โดยไม่ใส่ `Target` ใน prompt |
| `Publisher` | มาจาก `post_user` เมื่อมี โดยตัดเหลือไม่เกิน 120 ตัวอักษร |
| `Source URL` | มาจาก `feed_link` เมื่อเป็น HTTP(S) URL ที่อ่านได้ โดยส่งเฉพาะ hostname และ path ไม่เกิน 180 ตัวอักษรของ path; ไม่ส่ง query/fragment และไม่เปิดเว็บ |
| `Text` | ข้อความโพสต์หลังตัด HTML/entity และจัดช่องว่าง แล้วเลือกส่วนต้น/ท้ายและบริเวณรอบคีย์เวิร์ด/Target; Jev ค่าเริ่มต้น 1,800 ตัวอักษร ปรับได้ด้วย `JEV_TEXT_MAX_CHARS` (600–3,000) ส่วน DeepSeek ค่าเริ่มต้น 3,000 ตัวอักษร ปรับได้ด้วย `DEEPSEEK_TEXT_MAX_CHARS` สูงสุด 8,000 |

`Publisher` และ `Source URL` อาจปรากฏพร้อมกัน หรือไม่มีทั้งคู่ก็ได้ คีย์เวิร์ดใช้เลือกบริเวณ `Text` เท่านั้น ไม่ถูกใช้เป็น `Target`

## 2. Jev

ส่งไปที่ OpenRouter Decisions API ด้วย `model=typesafe/jev-1.13` ตามค่าเริ่มต้น โดยมี `state` และ `questions` ดังนี้

### `state`

```text
Target={Target}
Publisher={post_user ถ้ามี}
Source URL={hostname}{path ถ้ามี feed_link ที่ใช้ได้}
Text={ข้อความที่ตัดสำหรับ Jev}
```

บรรทัด `Publisher` และ `Source URL` เป็นทางเลือก ไม่มีบรรทัดว่างชดเชยเมื่อข้อมูลนั้นไม่มี

### `questions`

```json
{
  "sentiment": {
    "type": "choice",
    "instructions": "Classify sentiment toward the named Target project only. Keywords only selected the excerpt. A general topic mention without a clear connection to Target is neutral. Official Target news, PR and seller self-praise are neutral; a clearly attributed independent opinion about Target still counts. Publisher and URL alone do not prove Target relevance or official ownership.",
    "criteria": {
      "positive": "Independent praise, satisfaction, or support explicitly directed at Target, including an attributed quote; favorable news or advertiser self-praise alone does not qualify.",
      "neutral": "No subjective evaluation of Target: factual news, job ads, PR, seller slogans, or emotion about another person, film, event, or operator at a named venue.",
      "negative": "Complaint, blame, or criticism explicitly directed at Target, including an attributed quote; a venue named only as a location does not inherit criticism of others.",
      "irony": "Sarcasm, mockery, cynical humor, satire, or backhanded praise."
    }
  },
  "entity_relevance": {
    "type": "choice",
    "instructions": "Is the text clearly about the named Target project? A keyword or general topic mention alone is insufficient. Publisher and URL are context, not proof of relevance.",
    "criteria": {
      "relevant": "The text names Target or clearly attributes the discussed subject to Target, including neutral news.",
      "unrelated": "Only the retrieval keyword or a general topic appears, with no clear link to Target.",
      "uncertain": "It is unclear whether the text refers to a Target keyword."
    }
  },
  "intent": {
    "type": "choice",
    "instructions": "Classify the main purpose of the text concerning Target. A rhetorical question inside a grievance is a complaint. Pure praise is information. If Target is unrelated, choose information; the caller will omit intent.",
    "criteria": {
      "complaint": "The main purpose is to criticize, blame, or complain about Target.",
      "information": "The main purpose is to report facts, news, promotion, or pure praise about Target.",
      "recommendation": "The main purpose is to suggest an improvement to Target or recommend Target to other people.",
      "enquiry": "The main purpose is to ask for information about Target."
    }
  }
}
```

เมื่อมี Target Jev ตอบคำถามทั้งสามในคำขอเดียว ค่าเริ่มต้น `AI_COST_MODE=low` จะรับผล Jev เพิ่มเมื่อ relevance ชัดว่าไม่เกี่ยวข้อง หรือเกี่ยวข้องและ confidence อย่างน้อย 0.50 โดยไม่มี conflict; เคสที่ยังไม่แน่ชัดส่ง DeepSeek หากตั้ง `AI_COST_MODE=standard` จะใช้เกณฑ์เดิม confidence อย่างน้อย 0.65 และไม่มี conflict

## 3. DeepSeek

ส่งไปที่ OpenRouter Chat Completions ด้วย `model=deepseek/deepseek-v4-flash-0731` ตามค่าเริ่มต้น, `temperature=0`, และ `response_format={"type":"json_object"}`

### `system`

```text
Assess Thai social text toward the named Target project only. Keywords were used only to select the text excerpt. A keyword mention alone does not establish a connection to Target. A post about a broad topic in general is unrelated unless the text explicitly links that topic to Target. Emotion about another person, hardship, or event is neutral for Target unless it explicitly evaluates Target. A venue named only as the location of an event does not inherit criticism of the event or another person. Set entity_found=true only when the text names Target or clearly attributes the discussed subject to Target; otherwise set entity_found=false and NEUTRAL=1. Publisher and URL are context, not proof of relevance. Return JSON only: {"probabilities":{"POSITIVE":number,"NEUTRAL":number,"NEGATIVE":number,"AMBIGUOUS_OR_IRONY":number},"entity_found":boolean,"intent":string}. Probabilities must be finite, between 0 and 1, and sum to 1. POSITIVE=explicit independent praise, satisfaction, or support toward Target from a speaker, not the advertiser's own slogan; NEGATIVE=explicit complaint or criticism attributable to Target; hardship, crime news, or bad events merely mentioning Target are neutral; NEUTRAL=factual news, job ads, PR, ads, sales slogans, calls to buy, and the seller's own praise, even with words like 'love' or 'great'; for example 'Love X? Shop today' is an ad, not a consumer endorsement; AMBIGUOUS_OR_IRONY=sarcasm or mixed/unclear sentiment. Official Target posts and self-praise are neutral unless they clearly quote an independent person's opinion. For intent, classify the main purpose toward Target: complaint=grievance, information=facts or pure praise, recommendation=suggestion to improve Target or recommend Target to others, enquiry=request for information. A rhetorical question in a grievance is complaint. Omit intent when Target is unrelated. A preliminary Jev judgment may be included as a hint. Check the text independently, confirm the hint only when supported by the text, and correct it when the text disagrees. Do not mention Jev or the hint in the output.
```

### `user`

```text
Target={Target}
Publisher={post_user ถ้ามี}
Source URL={hostname}{path ถ้ามี feed_link ที่ใช้ได้}
Jev preliminary (hint; verify against text): sentiment={label}({confidence}); relevance={label}({confidence}); intent={label}; route_confidence={confidence}; conflicts={รายการ conflict หรือ none}
Text={ข้อความที่ตัดสำหรับ DeepSeek}
```

บรรทัด `Publisher` และ `Source URL` เป็นทางเลือกเช่นเดียวกับ Jev ส่วนบรรทัด `Jev preliminary` จะมีเมื่อ Jev ส่งผลที่ใช้ได้และ `DEEPSEEK_INCLUDE_JEV_SIGNAL` เปิดอยู่ (ค่าเริ่มต้น `true`) ค่า confidence แสดงสองตำแหน่งทศนิยม; field ที่ Jev ไม่มีจะแสดงเป็น `unknown` ผลนี้เป็น hint เท่านั้น และ DeepSeek ยังคงต้องส่ง JSON schema เดิมครบ

## 4. กรณีไม่มี Target: วิเคราะห์ภาพรวมของโพสต์

REST ยังใช้คีย์เวิร์ดเลือกบริเวณข้อความ แต่ไม่มี `Target` และไม่ถามว่าโพสต์เกี่ยวข้องกับโปรเจกต์ใด Jev ส่ง `state` รูปนี้:

```text
Scope=overall post sentiment and intent
Publisher={post_user ถ้ามี}
Source URL={hostname}{path ถ้ามี feed_link ที่ใช้ได้}
Text={ข้อความที่ตัดสำหรับ Jev}
```

Jev ส่ง `questions` สองข้อ:

```json
{
  "sentiment": {
    "type": "choice",
    "instructions": "Classify the overall sentiment expressed by the whole post. No named target exists; keywords only selected the excerpt. Factual news, PR and self-praise are neutral unless an independent opinion is clearly quoted.",
    "criteria": {
      "positive": "Praise, satisfaction, or support expressed in the post.",
      "neutral": "Factual reporting, PR, ads, or a question without a clear opinion.",
      "negative": "Complaint, blame, criticism, or dissatisfaction expressed in the post.",
      "irony": "Sarcasm, mockery, cynical humor, satire, or backhanded praise."
    }
  },
  "intent": {
    "type": "choice",
    "instructions": "Classify the main purpose of the whole post. A rhetorical question inside a grievance is a complaint. Pure praise is information.",
    "criteria": {
      "complaint": "The main purpose is to criticize, blame, or complain.",
      "information": "The main purpose is to report facts, news, promotion, or pure praise.",
      "recommendation": "The main purpose is to suggest an improvement or recommend something to other people.",
      "enquiry": "The main purpose is to ask for information."
    }
  }
}
```

ถ้า Jev ต้องส่งต่อ DeepSeek จะใช้ `system` นี้:

```text
Assess the overall sentiment and main intent of the whole Thai social post. There is no named target; keywords only selected the excerpt and are not the subject. Return JSON only: {"probabilities":{"POSITIVE":number,"NEUTRAL":number,"NEGATIVE":number,"AMBIGUOUS_OR_IRONY":number},"entity_found":true,"intent":string}. Probabilities must be finite, between 0 and 1, and sum to 1. POSITIVE=overall praise or satisfaction; NEGATIVE=overall complaint or criticism; NEUTRAL=facts, news, PR, ads, or questions without a clear opinion; AMBIGUOUS_OR_IRONY=sarcasm or mixed/unclear sentiment. Official self-praise and sales slogans are neutral unless an independent opinion is clearly quoted. Intent is the main purpose of the post: complaint, information, recommendation, or enquiry. Pure praise is information; recommendation includes suggestions and advice to others. A Jev preliminary judgment is only a hint; verify it against the text.
```

`user` ที่ส่ง DeepSeek:

```text
Scope=overall post sentiment and intent
Publisher={post_user ถ้ามี}
Source URL={hostname}{path ถ้ามี feed_link ที่ใช้ได้}
Jev preliminary (hint; verify against text): sentiment={label}({confidence}); intent={label}; route_confidence={confidence}; conflicts={รายการ conflict หรือ none}
Text={ข้อความที่ตัดสำหรับ DeepSeek}
```

บรรทัดข้อมูลแหล่งข่าวและ Jev preliminary เป็นทางเลือกตามข้อมูลที่มี ระบบส่ง `entity_found=true` ภายในเพื่อให้ sentiment และ intent ของภาพรวมถูกใช้ได้ โดยไม่ถือว่าพบชื่อโปรเจกต์

## 5. จุดในโค้ดสำหรับตรวจเทียบ

- REST เตรียมโพสต์และตั้ง scope: `ai_sentiment.py` → `SentimentAPI.run`
- เลือก Target และตัด Text: `ai_sentiment.py` → `_analyze_single_post`, `cap_text`
- จัด Publisher/URL: `ai_sentiment.py` → `_source_context`, `_compact_context_lines`
- Jev payload: `ai_sentiment.py` → `_call_typesafe_jev`
- DeepSeek payload: `ai_sentiment.py` → `KEYWORD_SYSTEM_PROMPT`, `build_deepseek_user_prompt`, `_call_deepseek_fallback`
