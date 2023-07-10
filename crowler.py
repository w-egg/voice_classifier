from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler


def crawel():
    print("画像クローリングを開始します。")
    print("取得する対象の名称を入力してください。(例:高野麻里佳)")
    print(">> ")

    search_word = input()

    print("取得する画像の枚数を入力してください。")
    get_num = input()
    if get_num.isdecimal() == True:
        get_num = int(get_num)
    else:
        print("数値が入力されなかったため処理を中止します。")
        return

    print("保存するディレクトリ名を入力してください。 ※アプリ配下のディレクトリに生成されます。")
    print(">> ")
    dir_name = input()

    crawel_auto(search_word, get_num, dir_name)


def crawel_auto(search_word, get_num, dir_name):
    print("Googleのクローリングを開始しました。")
    # Google
    googleCrawler = GoogleImageCrawler(storage={"root_dir": f'{dir_name}/'})
    googleCrawler.crawl(keyword=search_word, max_num=get_num)

    # print("Baiduのクローリングを開始しました。")
    # Baidu
    #baiduCrawler = BaiduImageCrawler(storage={"root_dir": f'{dir_name}/baidu'})
    #baiduCrawler.crawl(keyword=search_word, max_num=get_num)

    # print("Bingのクローリングを開始しました。")
    # Bing
    #bingCrawler = BingImageCrawler(storage={"root_dir": f'{dir_name}/bing'})
    #bingCrawler.crawl(keyword=search_word, max_num=get_num)


if __name__ == '__main__':
    crawel()
