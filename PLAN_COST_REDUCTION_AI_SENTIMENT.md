# แผนปรับปรุง: ลดต้นทุน AI Sentiment Pipeline (Jev + DeepSeek)

เอกสารนี้เป็น implementation plan สำหรับ `ai_sentiment.py` โดยอ้างอิงจากการรีวิว
`PROMPTS_REST_CURRENT.md` เป้าหมายคือลดต้นทุนต่อโพสต์ให้ได้มากที่สุด **โดยไม่ลด
accuracy ของผลลัพธ์ sentiment/entity_relevance/intent** ให้ทำตามลำดับหัวข้อ
1 → 6 เพราะเรียงตามผลกระทบต่อต้นทุนจากมากไปน้อย แต่ละหัวข้อทำแยกกันได้และควร
วัดผล (accuracy + cost) ก่อนไปข้อถัดไป

---

## หลักการก่อนเริ่ม

- **ห้ามแก้ threshold/logic แบบไม่วัดผล** ทุกการเปลี่ยนแปลงต้องมี before/after
  metric อย่างน้อย 2 ตัว: (a) % ของโพสต์ที่ escalate ไป DeepSeek (b) accuracy
  เทียบ ground-truth sample เดิม (ถ้ามี) หรือ agreement rate ระหว่าง Jev/DeepSeek
- เก็บ log ทุกครั้งที่ทดลอง (input tokens, output tokens, escalation flag) เพื่อ
  คำนวณ cost ต่อโพสต์จริง ไม่ใช่ประมาณ

---

## ขั้นที่ 0: วัด baseline ก่อนแก้อะไรทั้งสิ้น

**เป้าหมาย:** รู้ตัวเลขจริงก่อนตัดสินใจว่าควรลงแรงกับจุดไหนก่อน

- [ ] ดึง log ย้อนหลัง (อย่างน้อย 3,000–5,000 โพสต์ หรือ 7 วัน) นับ:
  - จำนวนโพสต์ทั้งหมดที่เข้า `SentimentAPI.run`
  - จำนวน/สัดส่วนที่ต้อง escalate ไป `_call_deepseek_fallback`
  - ค่าเฉลี่ยความยาว Text ที่ถูกส่งจริง (ทั้ง Jev และ DeepSeek) เทียบกับ cap
  - ค่า `DEEPSEEK_TEXT_MAX_CHARS` ที่ตั้งจริงใน production (ไม่ใช่ default ในโค้ด)
  - จำนวนโพสต์ที่ Target fallback ไปใช้คีย์เวิร์ด (ไม่มี project_name/actual_target/company_name)
- [ ] คำนวณ cost ปัจจุบันต่อโพสต์ (input+output token ทั้ง Jev และ DeepSeek เฉลี่ย)
      เพื่อใช้เป็น baseline เทียบหลังแก้แต่ละข้อ

**Output ของขั้นนี้:** ตารางตัวเลข baseline 1 ชุด เก็บไว้เทียบทุกขั้นถัดไป

---

## ขั้นที่ 1: ส่งผล Jev เข้า DeepSeek prompt (ผลกระทบสูงสุด)

**ปัญหา:** ทุกครั้งที่ escalate ไป DeepSeek ต้องวิเคราะห์ sentiment,
entity_relevance, intent ใหม่ทั้งหมด ทั้งที่ Jev วิเคราะห์มาแล้วรอบหนึ่ง

**สิ่งที่ต้องแก้ใน `_call_deepseek_fallback` / `build_deepseek_user_prompt`:**

1. เพิ่มบรรทัด Jev result เข้าไปใน `user` prompt เช่น:
   ```text
   Target={Target}
   Publisher={post_user ถ้ามี}
   Source URL={hostname}{path}
   Jev preliminary: sentiment={label}({confidence}), entity_relevance={label}, intent={label}
   Text={ข้อความที่ตัดสำหรับ DeepSeek}
   ```
   - ใส่เฉพาะ field ที่ Jev ตอบมาจริง (บาง field อาจไม่มีถ้า schema ไม่ครบ)
   - ต้องระบุ confidence/score ด้วย ไม่ใช่แค่ label เฉยๆ เพื่อให้ DeepSeek รู้ว่า
     "ควรเชื่อแค่ไหน"

2. ปรับ `KEYWORD_SYSTEM_PROMPT` เพิ่มประโยคทำนองนี้ (ต่อท้ายของเดิม ไม่ต้องเขียนใหม่ทั้งหมด):
   ```text
   A preliminary judgment from another model may be provided as "Jev preliminary".
   Treat it as a hint, not ground truth: confirm it only if the text supports it,
   and override it when your own reading of the text disagrees. Do not mention
   Jev or the hint in your output.
   ```

3. **Optional (ผลกระทบเพิ่มอีก):** ถ้า field ไหนที่ Jev ตอบมาแล้วและ "ผ่านเกณฑ์
   ชัดเจน" (เช่น entity_relevance=relevant ที่ confidence สูง) ให้ข้ามไม่ต้อง
   ขอ DeepSeek ตอบ field นั้นซ้ำ — ปรับ schema คำตอบ DeepSeek ให้ยืดหยุ่น
   (field เป็น optional เมื่อ Jev ยืนยันแล้ว) วิธีนี้ลด output token ได้ตรงจุด
   แต่ต้องแก้ parser ฝั่งรับผลด้วย ระวัง backward compatibility

**วิธีวัดผล:** เทียบ output token เฉลี่ยของ DeepSeek call ก่อน/หลัง +
agreement rate ระหว่างผล DeepSeek กับ Jev preliminary (ถ้า override บ่อยผิดปกติ
แปลว่า hint ทำให้ bias ผิดทาง ต้องปรับคำสั่งใน system prompt)

---

## ขั้นที่ 2: ตรวจและลด `DEEPSEEK_TEXT_MAX_CHARS`

**สิ่งที่ต้องทำ:**

1. หาค่าจริงที่ตั้งใน production env (ไม่ใช่ default 3,000 ในโค้ด)
2. Sample โพสต์จริง 200–300 โพสต์ วัดความยาว Text ที่ `cap_text` เลือกจริง
   ก่อนตัด แล้วดูว่ากี่ % ของโพสต์ยาวเกิน 3,000 / 5,000 / 8,000 ตัวอักษร
3. รัน DeepSeek ทดสอบ A/B ที่ cap 3,000 vs ค่าปัจจุบัน กับ sample เดียวกัน
   เทียบ accuracy (sentiment/entity_relevance/intent ตรงกันไหม)
4. ถ้า accuracy ไม่ต่างอย่างมีนัยสำคัญ → ลด `DEEPSEEK_TEXT_MAX_CHARS` ลงมาที่ค่า
   ต่ำสุดที่ accuracy ยังคงเดิม (คาดว่าจะอยู่แถว 3,000–4,000)

**หมายเหตุ:** เนื่องจาก `cap_text` เลือกส่วนต้น/ท้าย + รอบคีย์เวิร์ด/Target
อยู่แล้ว การเพิ่ม cap มักไม่ช่วย accuracy มากเท่าที่คิด เพราะเนื้อหาที่สำคัญ
ถูกเลือกไว้ตั้งแต่ต้น

---

## ขั้นที่ 3: แก้ Target fallback ไม่ให้ใช้คีย์เวิร์ดเปล่าๆ

**ปัญหา:** เมื่อไม่มี `project_name`/`actual_target`/`company_name` ระบบใช้
คีย์เวิร์ดเป็น Target ทำให้ `entity_relevance` มักตอบ "relevant" เกือบเสมอ
(เพราะคีย์เวิร์ดอยู่ในข้อความแน่นอน) แต่ความมั่นใจของ sentiment/intent มักต่ำ
เพราะ Target ไม่ใช่ entity จริง → เข้าเงื่อนไข escalate บ่อยขึ้นโดยไม่จำเป็น

**สิ่งที่ต้องแก้ใน `_analyze_single_post`:**

1. เพิ่มเงื่อนไข: ถ้าไม่มี `project_name`/`actual_target`/`company_name` เลย
   (เหลือแค่คีย์เวิร์ด) ให้เลือกหนึ่งในสองทาง (ต้องตัดสินใจร่วมกับทีม):
   - **ทางเลือก A (ประหยัดสุด):** ไม่เรียก AI เลย ตั้งผลลัพธ์เป็น
     `uncertain`/`skip` ทันทีด้วย heuristic
   - **ทางเลือก B (ยังอยาก analyze):** เรียกแค่ Jev อย่างเดียว ห้าม escalate
     ไป DeepSeek แม้ confidence ต่ำ (เพราะรู้อยู่แล้วว่าสาเหตุคือ data ไม่ครบ
     ไม่ใช่ความกำกวมของเนื้อหา)
2. Log แยกหมวดนี้ต่างหาก เพื่อดูสัดส่วนว่ากระทบ % ของ escalation รวมมากแค่ไหน

**วิธีวัดผล:** เทียบสัดส่วน escalation รวมก่อน/หลัง แยกเฉพาะกลุ่มที่ Target
มาจากคีย์เวิร์ด

---

## ขั้นที่ 4: เปิด prompt caching สำหรับ DeepSeek system prompt

**สิ่งที่ต้องทำ:**

1. เช็คว่า provider ที่ใช้ผ่าน OpenRouter สำหรับ `deepseek/deepseek-v4-flash-0731`
   รองรับ prompt caching หรือไม่ (เช็ค docs ของ OpenRouter/DeepSeek ปัจจุบัน
   เพราะอาจเปลี่ยนแปลงได้)
2. ถ้ารองรับ ปรับการเรียก API ให้ mark system prompt (`KEYWORD_SYSTEM_PROMPT`)
   เป็น cacheable block ตาม format ที่ provider กำหนด
3. วัด cost ต่อ call ก่อน/หลังเปิด caching (สังเกตว่า input token cost ของ
   system prompt ควรลดลงหลัง cache hit รอบถัดไป)

---

## ขั้นที่ 5: Pre-filter ก่อนเข้า AI (heuristic filter)

**เป้าหมาย:** กรองโพสต์ที่ "เป็น neutral โดยนิยามอยู่แล้ว" ออกก่อนเรียก Jev
เพื่อประหยัดทั้ง Jev และ DeepSeek cost

**สิ่งที่ต้องทำ:**

1. รวบรวม pattern ที่ system prompt กำหนดว่าเป็น neutral แน่นอน เช่น:
   - โฆษณาขาย/สโลแกนผู้ขาย ("Love X? Shop today" style)
   - ข่าว PR/ประกาศทางการที่ไม่มีการประเมินเชิงความเห็น
   - ประกาศรับสมัครงาน (job ads)
2. เขียน heuristic filter แบบ regex/keyword matching (เช่น มีลิงก์ร้านค้า +
   คำเชิญชวนซื้อ + ไม่มีคำแสดงความเห็นส่วนตัว) — ทำเป็น allowlist แคบๆ ก่อน
   ไม่ควร aggressive เกินไป เพราะ false positive จะทำให้เสีย accuracy จริง
3. Sample ตรวจ manual ก่อน deploy จริง (อย่างน้อย 100 โพสต์ที่ filter จับ)
   ยืนยันว่าไม่มีเคสที่ควรเป็น positive/negative หลุดเข้า filter นี้
4. เริ่มจาก filter แคบและ conservative ก่อน ค่อยขยายทีหลังเมื่อมั่นใจ

**คำเตือน:** ข้อนี้เสี่ยงกระทบ accuracy มากที่สุดในแผนทั้งหมด ให้ทำเป็นลำดับ
หลังสุด และวัดผลอย่างระมัดระวังกว่าข้ออื่น

---

## ขั้นที่ 6: Dedupe ก่อนวิเคราะห์

**สิ่งที่ต้องทำ:**

1. สร้าง cache key จาก hash ของ `(Target, Text ที่ตัดแล้วสำหรับวิเคราะห์)`
2. ก่อนเรียก Jev เช็ค cache ก่อน ถ้ามีผลลัพธ์เดิมอยู่แล้ว (ภายในช่วงเวลาที่
   กำหนด เช่น 24–72 ชม.) ให้ใช้ผลเดิมแทนการเรียก AI ใหม่
3. เลือก storage สำหรับ cache (Redis/DB ตาม stack ที่มีอยู่แล้วในระบบ)
4. กำหนด TTL ที่เหมาะสม — ไม่ควรนานเกินไปเพราะบริบทรอบข่าวอาจเปลี่ยน

---

## สรุปลำดับความคุ้มค่า (ทำก่อน-หลัง)

| ลำดับ | เรื่อง | ผลกระทบต้นทุนคาดการณ์ | ความเสี่ยงต่อ accuracy |
|---|---|---|---|
| 0 | วัด baseline | - | ไม่มี (ต้องทำก่อนเสมอ) |
| 1 | ส่ง Jev result เข้า DeepSeek | สูงมาก | ต่ำ (ถ้าเขียน prompt ดี) |
| 2 | ลด `DEEPSEEK_TEXT_MAX_CHARS` | สูง | ต่ำ-กลาง (ต้อง A/B ก่อน) |
| 3 | แก้ Target fallback (คีย์เวิร์ด) | กลาง-สูง | ต่ำ |
| 4 | Prompt caching | กลาง | ไม่มี |
| 5 | Pre-filter heuristic | กลาง | **สูง** (ต้องระวังที่สุด) |
| 6 | Dedupe | ต่ำ-กลาง (ขึ้นกับอัตราโพสต์ซ้ำจริง) | ไม่มี |

**คำแนะนำ:** ทำ 0 → 1 → 2 → 3 ก่อน แล้ววัดผลรวมอีกครั้งว่าลดต้นทุนได้กี่ %
ก่อนตัดสินใจว่าจะลงแรงกับข้อ 4–6 ต่อหรือไม่ เพราะข้อ 1–3 น่าจะให้ผลตอบแทนสูง
สุดเทียบกับความเสี่ยง
