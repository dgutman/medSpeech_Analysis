##hello
from dotenv import load_dotenv
## You need to import this before starting pixeltable.
load_dotenv()
import pixeltable as pxt


## Simple helper script to get the status of the message load

t = pxt.get_table('hani89_asr_data.transcribe_compare')

print(t.show(10).to_pandas())
print(t.count())