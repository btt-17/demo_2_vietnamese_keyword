from flask import Flask, jsonify, render_template, request
from vncorenlp import VnCoreNLP
from multiprocessing import Pool, Process, cpu_count
import urllib.request
from bs4 import BeautifulSoup

app = Flask(__name__,static_url_path="/static")

file = open('idf_dict.txt', 'r')
lines_in_files = file.readlines() 
rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
stop_words = ["là", "nên", "nếu", "thì", ",","."," nên","cạnh","bên","'","`",
                 "nêu", "năm ", "năm", "nói","\"", "”","“","...","'",":","?","-","_","diện",";",
                 "nén", "nâng", "bung", "ở","nước", "là", "sẽ","..","....",
                 "này", "của","đang","toàn", "từng","nhưng","lại",
                 "bị","-","các","chúng", "tôi", "ta", "bạn","thẳng","chung","gửi",
                 "tái", "thứ","tổ","để","ngày","học","lần","lấy","từ","đến","được",
                 "bay","giấy","lên","phải","có","đưa","việc","ông","bà","vào",
                 "họ","ở","lại","nếu","người","chủ","nhằm","tới","nhất","trước",
                 "nền","giữa","giới","cho hay","muốn","với","biết","về","cho",
                 "tuần","hồi","khiến","vụ","hòn","chuyến","kẻ","[","]","tắt","còn",
                 "hay","gọi","những","theo","nó","-","lâu","bài","cách","đây",
                 "không","một","viết","mang","nào","cũng","và","chuyển","tìm",
                 "đấu","áp","lề","cựu","trận","giải","sát","nơi","lời","dẫn",
                 "bản","…","kỳ","tháng","sau","đội","chục","sức","trong",
                 "cảnh","xem","quá","đều","nhau","vì","cuộc","dạng","nói chung",
                 "hàng","sắp","xảy","dậy","tưởng","mối","sự","cái","hỏi","siêu",
                 "(",")","đã","/","rằng","thuộc","mới","trên","cần","thấu",
                 "thạo","hậu","hôm","vững","nhắc","nhận","đứa","dựa","bàn","đứng",
                 "thành","hết","đệ","kể cả","dù","vẫn","cú","kể","mức","mà","ra",
                 "so","đó","hơn","như","vài","do","tại","diễn","thêm","số","thấy",
                 "gần","máy","vài","mọi","giá","số","sàn","bởi","đóng","khi","làm",
                 "câu","đặt","buổi","ai","thay","nay","nghĩ","qua","tay","hai",
                 "mỗi","gì","đạt","*","+","-","@","(",")","[","]","{","}","chi",":",
                 "tàn","vui","rồi","lúc","a","b","c","d","e","f","g","h","i","k","l",
                 "m","n","o","p","u","i","y","t","z","u","w","q","r","=","~","!","v",
                 "đ","ê","ă","â","ô","ơ","cả","lúc","gồm","rạng","à","á","ơi","hãy","lẽ",
                 "tối","sáng","ban","lan","sang","khác","chỉ","nhiều","ít","chào","bước",
                 "chúng ta","chúng tôi","điều","tấm","gương","giờ","giây","phút","chưa",
                 "sự việc","đơn","khuyên","mời","nóng","đốt","gây","mắc","tuổi",
                 "gấp","xa","bỏ","đơn","mời","hô","vang","lấp","trống","đầy","ngắt","lời",
                 "chơi","nhờ","lọt","xuống","chừng","dẫu sao","vất","khuôn","thì thầm","thẳng tay",
                 "ư","phọc","phun","ngậm","ướp","nhìn","rất","nhìn","sống","mình","rất","đi",
                 "tình trạng","sụt","tan","lún","chuốc","ko","cho dù","ý","sót","quả","sợ",
                 "lắm","mấy","anh","thật","nghe","gởi","gửi","thức","kỹ","đi","dễ","thiếu","mất",
                 "đủ","rất","ai ai","người người","nhà nhà","đăng","làm sao","bây giờ","nghỉ","tuyển",
                 "gặt","ngọt","thời","công","ăn","gặp","tiếng","dài","ngắn","phía","gặp","ném","thói","vv",
                 "tranh","đấy","chất","sai","một vài","noi","ngoài","cũng nên","cậu","trai","sân","to","dưới",
                 "lầu","ví dụ","vượt","nhanh","một cách","nhìn","nhìn chung","xẹt","xịt","bằng","lãi","vốn",
                 "buộc","tên","lớp","xướng","một","hai","ba","bốn","năm","sáu","bảy","tám","chín","mười",
                 "đổi","quỹ","vòng","treo","chở","nằm","ngồi","đứng","yếu","mặt","mạnh","nhanh chóng","chậm trễ",
                 "đòi","lửa","suốt","loạt","hướng","khó","dần","cỡ","tung","đa","quân","miền","dành","phiên",
                 "đáng","rà","phá","tin","trang","luôn","trải","hãng","ca","nhận định","phiếu","phát biểu",
                 "báo","bao gồm","vùng","chúng tôi","cấp","nhân dịp","trực thuộc","niềm","ạ","em","phép",
                 "nắm","quyền","news","bầu","trú","phường","có vẻ","vừa","tỉnh","đối với","tình trạng",
                 "cơ số","má","đợi","đời","khóc","bất cần","đồng","để","đại","với","từ","phối","phân","nhập",
                 "kí","vậy","đầu","tư","ở","hơn","thông","số","công","châu","chắc","thôi","xin","bứt","đôi",
                 "đuôi","đuối","điểm","trần","bật","cục","chịu","đối với","đảo","phi","nghiêm","giao","thu",
                 "hoặc","đối với","đẩy","bốc","kéo","chia","lớn","động","đơn vị","được","ưu","tư","tiên",
                 "thuật","tham chiếu","rộng","hẹp","vậy","vượt","hiệu","hướng","nên","hơn","công","phao",
                 "nổ","núi","ơn chúa","Rg","ra vào","nối","thoa","k.","làm bàn","tên riêng","sít sao",
                 "để chế","thu dung","sự việc thể","giúc","như ai","với","đã","hợp","trả","đồng","làm",
                 "phải","nỗ","giáng","đòn","bấy giờ","mở","mượt","chứng tỏ","gươm","lát","chốc","giả như",
                 "tíchthứ", "bự","tổ chảng","các","chặng","hạng","chiếc",]


@app.route('/') 
def homepage():
    return render_template("homepage.html")

@app.route('/', methods=['POST','GET'])
def get_result():
    input = format(request.form['text'])
    default_key= {"Từ khoá 1":"0.0", "Từ khoá 2":"0.0","Từ khoá 3":"0.0","Từ khoá 4":"0.0",
                    "Từ khoá 5":"0.0","Từ khoá 6":"0.0"}
    if request.form.get('nut2', None) == "delete":
        return render_template("result.html", content = "", keywords = default_key)

    #title = title_extract(input)
    #content = content_extract(input)
    
    if input == "":
        return render_template("homepage.html")
    content = input.replace("\n","")
    keywords = process(content) 
    result = ""
    list_key = list(keywords.keys())
    for i in range(len(list_key)):
        result += list_key[i] +","
    
    if request.form.get('nut1', None) == "find":
        return render_template("result.html", content = content.lower(), keywords = keywords,result=result)
    return render_template("result.html", content = content.lower(), keywords = keywords,result=result)

def content_extract(input):
    webUrl = urllib.request.urlopen(input)
    html = webUrl.read()    
    soup = BeautifulSoup(html, features="html.parser")
    content = soup.findAll('p',class_="Normal")
    result = ""
    for elm in content:
        result += elm.text + " "
    return result

def title_extract(input):
    webUrl = urllib.request.urlopen(input)
    html = webUrl.read()    
    soup = BeautifulSoup(html, features="html.parser")
    title = soup.title.string
    return title

def removeSpecialCharacter(text):
    #text = text.replace(".", ". ")
    #text = text.replace(",", " ,")
    text = text.replace(":", " ")
    text = text.replace("?", " ")
    text = text.replace("\""," ")
    text = text.replace("!", "")
    text = text.replace("@", " ")
    text = text.replace("'", " ")
    text = text.replace("`", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("-", " ")
    text = text.lower()
    return text

def process(data):
    
    #data = removeSpecialCharacter(data)
    sentences = rdrsegmenter.tokenize(data)  

    set_text = set()
    for i in range(len(sentences)):
        set_add = set(sentences[i])
        for elm in set_add:
            set_text.add(elm)

    #title = removeSpecialCharacter(title)

    def cal_score(arg_str):
        title_score = 0
        idf = 8
        count = sentences[0].count(arg_str)

        word = arg_str.lower()
        for line in lines_in_files:
            new_line = line.replace("\n","")
            line_word = new_line.split(" ")[0]
            if line_word == word:
                idf = float(new_line.split(" ")[1])
                break;
        #if arg_str.replace("_", " ") in title:
        #title_score = 10
        if (arg_str.replace(".","").isdigit() == True) or (arg_str.replace(",","").isdigit() == True):
            idf = 5

        tf = count/len(sentences[0])
        score = (tf+1)*(idf)

        return str(score)+ " "+ arg_str.replace("_", " ") 
    
    #process_pool = Pool(processes=cpu_count())
    #output = process_pool.map(cal_score,set_text)
    #return output

    output_list = []
    for word in set_text:
        word2 = word.lower()
        word2 = word2.replace(" ","")
        word2 = word2.replace("_"," ")   
        if (word2 not in stop_words) and (word2.replace(".","") not in stop_words ):
            output_list.append(cal_score(word))

    output_list.sort(reverse=True)


    #print(output_list)
    output = {}
    output_len = len(output_list)
    threshold = 0
    if output_len > 15:
        threshold = 15
    else:
        threshold = output_len
    for i in range(threshold):
        word =  ""
        word_list = output_list[i].split(" ")
        for j in range(1,len(word_list)):
            word += word_list[j] + " "
        add_map = {word.lower():word_list[0]}
        output.update(add_map)
    
    return output
    

if __name__ == '__main__':
    #app.run(threaded=True,debug=True)
    app.run(host="localhost", port=8006, debug=False)