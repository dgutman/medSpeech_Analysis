from dotenv import load_dotenv
import pixeltable as pxt
load_dotenv()



t = pxt.get_table('hani89_asr_data.transcribe_compare')

#t.push()
# #t.count()
# print(t.count())
# # Publish to cloud (make it shareable)
pxt.publish(t, "pxt://speech-to-text-analytics/hani89_asr_dataset_v1", access='public')
# # ## So how does versioning/latest_work