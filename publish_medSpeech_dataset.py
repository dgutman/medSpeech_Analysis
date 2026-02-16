import os
from dotenv import load_dotenv
load_dotenv()

import pixeltable as pxt

# Get table configuration from .env with defaults
pxt_dir = os.getenv('PIXELTABLE_DIR', 'hani89_asr_data_reload')
table_name = os.getenv('PIXELTABLE_TABLE', 'transcribe_compare')
full_table_name = f'{pxt_dir}.{table_name}'

t = pxt.get_table(full_table_name)


print(t.count(),"rows in the table")
# # Publish to cloud (make it shareable)
# Get publish path from .env with default
## We have to publish this first, and then we can push..
publish_path = os.getenv('PIXELTABLE_PUBLISH_PATH', 'pxt://speech-to-text-analytics/hani89_asr_data_reload.transcribe_compare')
#pxt.publish(t, publish_path, access='public')
# # ## So how does versioning/latest_work
t.push()
