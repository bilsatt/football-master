# Football Master — Cloud Deploy (Yol 1)

## Dosyalar
- `apifootball_master_v5c_cloud.py`  ← app file
- `model_plus_dc.py`                 ← aynı klasöre koy
- `requirements.txt`                 ← pip paketleri
- `secrets.example.toml`             ← örnek secrets

## Streamlit Community Cloud
1) Kodu GitHub'a push et.
2) https://share.streamlit.io → **New app** → repo/branch/app file = `apifootball_master_v5c_cloud.py`
3) App → **Settings → Secrets** → şu TOML'u yapıştır:
   ```toml
   APIFOOTBALL_KEY = "your_apifootball_key_here"
   ODDALERTS_TOKEN = "your_oddalerts_token_here"
   ```
4) Deploy.
5) Telefonda URL'yi Chrome'da aç → menü → **Ana ekrana ekle**.

## Hugging Face Spaces (alternatif)
1) New Space → SDK: **Streamlit**
2) Files: `apifootball_master_v5c_cloud.py`, `model_plus_dc.py`, `requirements.txt`
3) Space Settings → **Variables/Secrets** → `APIFOOTBALL_KEY`, `ODDALERTS_TOKEN`
4) App file: `apifootball_master_v5c_cloud.py`

## Yerel test (opsiyonel)
```bash
pip install -r requirements.txt
# yerelde secrets için:
set APIFOOTBALL_KEY=xxx   # PowerShell: $env:APIFOOTBALL_KEY="xxx"
set ODDALERTS_TOKEN=yyy
streamlit run apifootball_master_v5c_cloud.py
```
