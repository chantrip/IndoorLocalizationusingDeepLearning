@ECHO OFF
for /l %%x in (1, 1, 10) do (
rem (python client.py) > Output 
python fix_test.py
rem SET /p MYVAR=<Output
rem ECHO %MYVAR%	
ECHO "=========================================================="   
)
pause