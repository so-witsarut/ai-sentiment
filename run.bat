@echo off
chcp 65001 > nul
title AI Sentiment Analysis — Running Forever

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║     AI Sentiment Analysis — Windows Runner      ║
echo  ╚══════════════════════════════════════════════════╝
echo.
echo  กด Ctrl+C เพื่อหยุดการทำงาน
echo.

rem ตรวจสอบว่ามี .venv อยู่ไหม
if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] ไม่พบ .venv — กรุณารัน setup.bat ก่อน
    pause
    exit /b 1
)

rem ตรวจสอบว่ามี .env อยู่ไหม
if not exist ".env" (
    echo  [WARNING] ไม่พบไฟล์ .env — คัดลอก .env.example เป็น .env แล้วใส่ credentials
    echo  ใช้โหมด mockup แทน...
    echo.
)

rem รันสคริปต์
.venv\Scripts\python.exe run_forever.py

echo.
echo  [หยุดทำงานแล้ว]
pause
