# 下載影片音檔
from pytube import YouTube # 使用前要先 pip install pytube 
link = input('Enter URL:') # 使用輸入介面，不用每次進來改 code
print('Downloading...') 
yt = YouTube(link) # 把影片傳入並且命名為 yt 
yt.streams.filter(only_audio=True).first().download(filename='video.mp4') # Download Audio：最難搞的一段，一定要有 first() 
print('Audio : "' + yt.title + '" downloaded as file `vedio.mp4`')

# MP4 轉 MP3
import os 
cmd = "ffmpeg -i {} -vn {}".format("video.mp4", "audio.mp3") # Mp4 轉 Mp3 ，使用 ffmpeg
os.system(cmd) # 直接把指令送進去終端機去操作
os.remove("video.mp4") # 轉完音檔舊檔案就可以刪了
print('mp4 to mp3: Done!')

# 音檔轉錄
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
text = result["text"]                         # Transcription here
script = open("script.txt", 'a')
script.write(text)
print("Transcription done!")

# 把長文檔切分成小塊，回傳一個 chunks 數列（List）
def split_text(text):
    """Splitting text base on max_chunk_size"""
    max_chunk_size = 16000   
    chunks = [] 
    current_chunk = "" 
    for sentence in text.split("."):     
        if len(current_chunk) + len(sentence) < max_chunk_size: 
            current_chunk += sentence + "." 
        else:   
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "." 
    if current_chunk: 
        chunks.append(current_chunk.strip()) 
    return chunks   

# 呼叫 openai API，將 chunks 送進去 GPT-3.5 模組生成總結
import openai
import tiktoken
import constant
openai.api_key = constant.api_key 
def generate_summary(text):
    """Generate summary by while-looping chunks of text to GPT-3.5 to form summarization"""
    token_total = 0
    while len(tiktoken.get_encoding("gpt2").encode(text)) > 4090:    # Get token length with tiktoken
        input_chunks = split_text(text)   # Split text by 16000 characters
        print("Summarize start, there are {} chunks to summarize.".format(len(input_chunks)))
        output_chunks = []
        for i, chunk in enumerate(input_chunks):
            token_count = len(tiktoken.get_encoding("gpt2").encode(chunk))   # count token usage
            print("Summarizing, token used in this round: {}".format(token_count))
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[ 
                {"role": "system", "content": "you are a helpful assistant who provides deep insight, also a passive-agressive, guilt-tripping Jewish mother."}, 
                {"role": "user", "content": "Your task is to analyze the text I give you and report the key points in bullets without losing any form of exact details. First add a Title to the report in h1 format, do not append the word `title`, there should be no limit in words to the report, ensure that all the points and details are comprehensively reported out.  Please use as many bullet points as needed. After reporting out Finally add a `Key Takeaway` from the text you just summarized. I want you to write in markdown format. The text should be read from here: {}".format(chunk)} ]
            )
            summary = response["choices"][0]["message"]["content"]
            output_chunks.append(summary)
            text = " ".join(output_chunks)  
            token_total += token_count
            print("chunk{0} summarized.".format(i))
    doc = open("Summary.txt", 'a')
    doc.write(text)
    doc.close()
    print("Summarization success! please check Summary.txt")
    print("Summarization Info:")
    print("    Token count: {}".format(token_total))
    print("    Cost: {} USD".format(token_total * 0.000002))
    
generate_summary(text)