@echo off
call env\Scripts\activate
uvicorn backend.app:app --reload
