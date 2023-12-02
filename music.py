import requests
import json

# 伪装自己的请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43"
}

# 音乐列表请求
list_url = "https://complexsearch.kugou.com/v2/search/song?callback" \
           "=callback123&srcappid=2919&clientver=1000&clienttime=" \
           "1686899474798&mid=a50def723fb336f1ef3549fe0156a513&uuid=" \
           "a50def723fb336f1ef3549fe0156a513&dfid=2C7EhC1s22Gg13W7sN0Mp" \
           "J8L&keyword=%E5%A4%A7%E9%B1%BC&page=1&pagesize=30&bitrate=" \
           "0&isfuzzy=0&inputtype=0&platform=WebFilter&userid=0&iscorrection=" \
           "1&privilege_filter=0&filter=10&token=&appid=1014&signature" \
           "=71f366a98842287d22e4ecd0a8639f1d"
list_resp = requests.get(list_url, headers=headers)
song_list = json.loads(list_resp.text[12:-2])['data']['lists']
for i, s in enumerate(song_list):
    print(f'{i+1}----{s.get("SongName")}----{s.get("EMixSongID")}')

num = input('请输入要下载第几首音乐：')
# 音乐信息的url地址
info_url = f'https://wwwapi.kugou.com/yy/index.php?r=play/getdata&encode_album_audio_id={song_list[int(num)-1].get("EMixSongID")}'
headers2 = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43",
    "Cookie": "kg_mid=a50def723fb336f1ef3549fe0156a513; kg_dfid=2C7EhC1s22Gg13W7sN0MpJ8L; kg_dfid_collect=d41d8cd98f00b204e9800998ecf8427e"
}
info_resp = requests.get(info_url, headers=headers2)
play_url = json.loads(info_resp.text)['data']['play_url']
print(info_url, play_url)
# 发送请求到服务器，获取音乐资源
m_resp = requests.get(play_url, headers=headers)
# 保存音乐文件
with open('zzz.mp3', 'wb') as f:
    f.write(m_resp.content)
