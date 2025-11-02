import pixeltable as pxt
from dotenv import load_dotenv
load_dotenv()

print(pxt.__version__,"version of pixeltable")
# Replicate this table to your local environment
local_table = pxt.replicate(
    remote_uri='pxt://speech-to-text-analytics:main/hani89_asr_dataset',
    local_path='local_hani89'
)

# Pull latest data
#local_table.pull()


