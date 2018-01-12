#!/usr/bin/env bash
python3 main.py supervised ../Data/Patches/Training/ ../Data/Patches/Testing/;

python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ add 1;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ add 5;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ add 10;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ add 50;

python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ add 1;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ add 5;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ add 10;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ add 50;

python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ add 1;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ add 5;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ add 10;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ add 50;


python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 1;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 5;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 10;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 50;

python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 1;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 5;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 10;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 50;

python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 1;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 5;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 10;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ replace 50;


python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 1;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 5;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 10;
python3 main.py bootstrap ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 50;

python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 1;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 5;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 10;
python3 main.py uncertainty ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 50;

python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 1;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 5;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 10;
python3 main.py random ../Data/Patches/Training/ ../Data/Patches/Testing/ merge 50;