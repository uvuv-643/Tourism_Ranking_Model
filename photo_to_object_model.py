from torch import nn
import torch
import torchvision.models as models
import time
import pickle
import base64
from PIL import Image
import io
from joblib import dump, load
import pandas as pd
import json

from utils import json_from_pandas_to_main_format


class PhotoModel:
    def __init__(self, n_classes=290):
        self.train_data = pd.read_excel('All_cities.xlsx')
        russian_kinds = {'accomodations': 'жилье',
                         'archaeological_museums': 'археологические музеи',
                         'archaeology': 'археология',
                         'architecture': 'архитектура',
                         'art_galleries': 'художественные галереи',
                         'bank': 'банк',
                         'banks': 'банки',
                         'biographical_museums': 'биографические музеи',
                         'bridges': 'мосты',
                         'burial_places': 'места захоронений',
                         'castles': 'замки',
                         'cathedrals': 'соборы',
                         'catholic_churches': 'католические церкви',
                         'cemeteries': 'кладбища',
                         'children_museums': 'детские музеи',
                         'children_theatres': 'детские театры',
                         'churches': 'церкви',
                         'cinemas': 'кинотеатры',
                         'circuses': 'цирки',
                         'concert_halls': 'концертные залы',
                         'cultural': 'культурный',
                         'defensive_walls': 'защитные стены',
                         'destroyed_objects': 'разрушенные объекты',
                         'eastern_orthodox_churches': 'восточные ортодоксальные церкви',
                         'fashion_museums': 'музеи моды',
                         'foods': 'продукты питания',
                         'fortifications': 'фортификационные сооружения',
                         'fortified_towers': 'укрепленные башни',
                         'fountains': 'фонтаны',
                         'gardens_and_parks': 'сады и парки',
                         'geological_formations': 'геологические образования',
                         'historic': 'исторические',
                         'historic_architecture': 'историческая архитектура',
                         'historic_districts': 'исторические районы',
                         'historic_house_museums': 'исторические дома музеи',
                         'historic_object': 'исторические объекты',
                         'historic_settlements': 'исторические поселения',
                         'historical_places': 'исторические места',
                         'history_museums': 'исторические музеи',
                         'industrial_facilities': 'промышленные объекты',
                         'installation': 'инсталляции',
                         'interesting_places': 'интересные места',
                         'kremlins': 'кремли',
                         'local_museums': 'местные музеи',
                         'manor_houses': 'усадьбы',
                         'military_museums': 'военные музеи',
                         'monasteries': 'монастыри',
                         'monuments': 'памятники',
                         'monuments_and_memorials': 'монументы и памятники',
                         'mosques': 'мечети',
                         'mountain_peaks': 'горные вершины',
                         'museums': 'музеи',
                         'museums_of_science_and_technology': 'музеи науки и технологии',
                         'music_venues': 'музыкальные места',
                         'national_museums': 'национальные музеи',
                         'natural': 'природные',
                         'natural_monuments': 'природные монументы',
                         'nature_reserves': 'природные заповедники',
                         'open_air_museums': 'музеи под открытым небом',
                         'opera_houses': 'оперные театры',
                         'other': 'другие',
                         'other_archaeological_sites': 'другие археологические объекты',
                         'other_bridges': 'другие мосты',
                         'other_buildings_and_structures': 'другие здания и сооружения',
                         'other_burial_places': 'другие места захоронения',
                         'other_churches': 'другие церкви',
                         'other_hotels': 'другие отели',
                         'other_museums': 'другие музеи',
                         'other_nature_conservation_areas': 'другие природоохранные зоны',
                         'other_technology_museums': 'другие технологические музеи',
                         'other_temples': 'другие храмы',
                         'other_theatres': 'другие театры',
                         'other_towers': 'другие башни',
                         'planetariums': 'планетарии',
                         'puppetries': 'кукольные театры',
                         'railway_stations': 'железнодорожные станции',
                         'religion': 'религия',
                         'restaurants': 'рестораны',
                         'rock_formations': 'скальные образования',
                         'science_museums': 'научные музеи',
                         'sculptures': 'скульптуры',
                         'sport': 'спорт',
                         'squares': 'площади',
                         'stadiums': 'стадионы',
                         'theatres_and_entertainments': 'театры и развлечения',
                         'tourist_facilities': 'туристические объекты',
                         'tourist_object': 'туристический объект',
                         'towers': 'башни',
                         'triumphal_archs': 'триумфальные арки',
                         'unclassified_objects': 'неклассифицированные объекты',
                         'urban_environment': 'городская среда',
                         'view_points': 'смотровые площадки',
                         'war_memorials': 'военные памятные места',
                         'water_towers': 'водяные башни',
                         'zoos': 'зоопарки',
                         'schools': 'школы',
                         'education': 'образование',
                         'universities': 'университеты'}
        self.train_data['Kind'] = self.train_data['Kind'].apply(lambda x: [russian_kinds[i] for i in x.split(',')])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = models.vit_l_16()
        n_classes = n_classes
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes),
        )
        self.transform = models.vision_transformer.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        self.model.heads = classifier
        self.model.load_state_dict(torch.load("models/photo_model.pt", map_location=device))
        print("model loaded")
        self.model = self.model.to(device)
        self.model.eval()
        self.inverse_transform = {0: '1941-1945',
                                  1: '2000 летию Рождества Христова',
                                  2: 'église arménienne de Vladimir',
                                  3: 'Авиамеханический колледж',
                                  4: 'Автозаводский парк культуры и отдыха',
                                  5: 'Администрация Нижнего Новгорода',
                                  6: 'Александро-Невская часовня',
                                  7: 'Александро-Невский собор',
                                  8: 'Аптекарь',
                                  9: 'Белая башня',
                                  10: 'Белокаменные памятники Владимира и Суздаля',
                                  11: 'Библиотечная лужайка',
                                  12: 'Благовещенский монастырь',
                                  13: 'Благовещенский собор',
                                  14: 'Блиновский пассаж',
                                  15: 'Бобёр',
                                  16: 'Богородице-Рождественский мужской монастырь',
                                  17: 'Борисоглебская башня',
                                  18: 'Бывшая водонапорная башня',
                                  19: 'Великая Княгиня Ольга',
                                  20: 'Венера',
                                  21: 'Вечный огонь',
                                  22: 'Виртуальный концертный зал',
                                  23: 'Владимир',
                                  24: 'Владимиро-Суздальский музей-заповедник (центральный офис)',
                                  25: 'Владимирская вишня',
                                  26: 'Владимирская гвардейская ракетная Витебская Краснознамённая армия',
                                  27: 'Владимирская духовная семинария',
                                  28: 'Владимирская митрополия',
                                  29: 'Владимирский академический областной драматический театр',
                                  30: 'Владимирский государственный гуманитарный университет',
                                  31: 'Владимирский областной театр кукол',
                                  32: 'Владимирский планетарий',
                                  33: 'Владимирский централ',
                                  34: 'Владимирским железнодорожникам погибшим в ВОВ',
                                  35: 'Владимирское духовное училище',
                                  36: 'Водонапорная башня',
                                  37: 'Военно-историческая экспозиция (Золотые ворота)',
                                  38: 'Воинам-интернационалистам',
                                  39: 'Второй Дом Советов',
                                  40: 'Выставочный зал',
                                  41: 'Галерея Владимирской школы живописи',
                                  42: 'Гарнизонная казарма',
                                  43: 'Георгиевская башня',
                                  44: 'Гимназия № 9',
                                  45: 'Городок Чекистов',
                                  46: 'Горьковчане фронту',
                                  47: 'Государственный центральный банк',
                                  48: 'Губернаторский сад',
                                  49: 'Дворец вице-губернатора',
                                  50: 'Дева Мария',
                                  51: 'Дерево дружбы',
                                  52: 'Динамо',
                                  53: 'Дмитриевская башня',
                                  54: 'Дмитриевский собор',
                                  55: 'Дмитровская башня',
                                  56: 'Добровольцам-танкистам',
                                  57: 'Дом А. И. Фролова',
                                  58: 'Дом А. К. Фомина',
                                  59: 'Дом Актёра',
                                  60: 'Дом Артистов',
                                  61: 'Дом Бревновых',
                                  62: 'Дом В. И. Смирнова',
                                  63: 'Дом Виноградовых',
                                  64: 'Дом Гайдара',
                                  65: 'Дом Е. Трушенинниковой',
                                  66: 'Дом Ермолаевых — П. Парфёнова',
                                  67: 'Дом И. Н. Соболева',
                                  68: 'Дом М. В. Медведева',
                                  69: 'Дом Маева',
                                  70: 'Дом Метенкова',
                                  71: 'Дом Промышленности',
                                  72: 'Дом Сироткина',
                                  73: 'Дом Советов',
                                  74: 'Дом Степанова',
                                  75: 'Дом Э. Ф. Филитц',
                                  76: 'Дом архитектора',
                                  77: 'Дом дворянина А.А. Протасьева',
                                  78: 'Дом дворянки Е. О. Селивановой',
                                  79: 'Дом контор',
                                  80: 'Дом культуры ГУВД (Нижний Новгород)',
                                  81: 'Дом обороны',
                                  82: 'Дом печати (Екатеринбург)',
                                  83: 'Дом связи (Екатеринбург)',
                                  84: 'Дом-музей Бажова',
                                  85: 'Дом-музей Д. Н. Мамина-Сибиряка',
                                  86: 'Дом-музей Добролюбова',
                                  87: 'Дом-музей Столетовых',
                                  88: 'Дом-музей Ф.М. Решетникова',
                                  89: 'Домик Каширина',
                                  90: 'Домик Петра I',
                                  91: 'Домовая церковь Елисаветы Феодоровны',
                                  92: 'Доходный дом Б. Н. Юсупова',
                                  93: 'Доходный дом купца Чувильдина',
                                  94: 'Доходный дом купчихи П. Е. Кубаревой',
                                  95: 'Дружба',
                                  96: 'Екатеринбургский завод',
                                  97: 'Екатеринбургский музей изобразительных искусств',
                                  98: 'Зачатьевская башня',
                                  99: 'Здание Волжско-Камского банка',
                                  100: 'Здание Присутственных мест. "Палаты"',
                                  101: 'Здание Удельной конторы',
                                  102: 'Здание бывшей гостиницы "Мадрид"',
                                  103: 'Здание городской думы',
                                  104: 'Здание пансиона и церкви Святой Магдалины при первой женской гимназии',
                                  105: 'Золотые ворота',
                                  106: 'Ивановская башня',
                                  107: 'Ивановский вал',
                                  108: 'Иоанно-Предтеченский Кафедральный собор',
                                  109: 'Исторический музей',
                                  110: 'Казанская церковь (Кстово)',
                                  111: 'Каменная горка',
                                  112: 'Каменный мост',
                                  113: 'Кино 9D',
                                  114: 'Кладовая башня',
                                  115: 'Князь-Владимирская церковь',
                                  116: 'Князь-Владимирское кладбище',
                                  117: 'Князю Владимиру и святителю Фёдору',
                                  118: 'Козлов вал',
                                  119: 'Комплекс штаба военного округа',
                                  120: 'Комсомолу Урала',
                                  121: 'Концертно-спортивный комплекс Art Hall',
                                  122: 'Концертный зал им. С. И. Танеева',
                                  123: 'Коромыслова башня',
                                  124: 'Костел Святого Розария Пресвятой Девы Марии Римско-Католической Церкви',
                                  125: 'Кресень',
                                  126: 'Крестовоздвиженская церковь (Екатеринбург)',
                                  127: 'Крестовоздвиженский монастырь',
                                  128: 'Крестовоздвиженский собор',
                                  129: 'Лебедев-Полянский П.И.',
                                  130: 'Ленин',
                                  131: 'Литературная жизнь Урала XX века',
                                  132: 'Лицей № 82',
                                  133: 'Локомотив',
                                  134: 'М. Горькому',
                                  135: 'Макет ракеты-носителя Союз-ТМ',
                                  136: 'Малышеву',
                                  137: 'Мальчик с рогаткой',
                                  138: 'Мемориал памяти Уральских коммунаров',
                                  139: 'Мемориальная мастерская им. Французова',
                                  140: 'Монумент героям Волжской военной флотилии',
                                  141: 'Музей «Литературная жизнь Урала XIX века»',
                                  142: 'Музей изобразительных искусств',
                                  143: 'Музей истории Екатеринбурга',
                                  144: 'Музей истории, науки и техники Свердловской железной дороги',
                                  145: 'Музей кукол и детской книги «Страна чудес»',
                                  146: 'Музей ложки',
                                  147: 'Музей науки «Нижегородская радиолаборатория»',
                                  148: 'Музей науки и человека ЭВРИКА, г. Владимир',
                                  149: 'Музей непридуманных историй',
                                  150: 'Музей плодового садоводства Среднего Урала',
                                  151: 'Музей природы',
                                  152: 'Музей хрусталя и стекла XVIII-XXI веков',
                                  153: 'Музей-квартира А.М. Горького',
                                  154: 'Н. И. Кузнецову',
                                  155: 'Надвратная церковь святого князя Александра Невского',
                                  156: 'Наука',
                                  157: 'Нижегородская государственная консерватория им. М.И. Глинки',
                                  158: 'Нижегородская соборная мечеть',
                                  159: 'Нижегородский Кремль',
                                  160: 'Нижегородский государственный академический театр драмы им. М. Горького',
                                  161: 'Нижегородский государственный художественный музей',
                                  162: 'Нижегородский кремль',
                                  163: 'Нижегородский острог',
                                  164: 'Нижегородское речное училище',
                                  165: 'Никольская башня',
                                  166: 'Ново-Тихвинский монастырь',
                                  167: 'Обелиск Минину и Пожарскому',
                                  168: 'Оборонительный земляной вал древнего Владимира',
                                  169: 'Огни Владимира',
                                  170: 'П.И. Чайковский',
                                  171: 'Памятник Гоголю',
                                  172: 'Памятник Ленину',
                                  173: 'Памятник Я.М. Свердлову',
                                  174: 'Памятник жертвам и мученикам революции 1905 года',
                                  175: 'Памятник природы "Лесной парк Дружба"',
                                  176: 'Панно на стене',
                                  177: 'Панно на фасаде',
                                  178: 'Парк имени А.С. Пушкина',
                                  179: 'Парк имени Кулибина',
                                  180: 'Печёрский Вознесенский монастырь',
                                  181: 'Площадка на крыше',
                                  182: 'Площадь 1-й Пятилетки',
                                  183: 'Площадь 1905 года',
                                  184: 'Погибшим в ВОВ',
                                  185: 'Пожарный автомобиль-лестница',
                                  186: 'Пороховая башня',
                                  187: 'Преображенская церковь на Уктусе',
                                  188: 'Притвор-часовня в честь иконы Божией Матери "Иверская"',
                                  189: 'Рекорд',
                                  190: 'Рождественская церковь',
                                  191: 'Свердловский областной краеведческий музей',
                                  192: 'Свердловское художественное училище им. И. Д. Шадра',
                                  193: 'Свято-Никольский храм (Екатеринбург)',
                                  194: 'Свято-Троицкий Кафедральный собор',
                                  195: 'Северная башня',
                                  196: 'Серго Орджоникидзе',
                                  197: 'Сердце',
                                  198: 'Скоба',
                                  199: 'Собор Александра Невского',
                                  200: 'Собор Архангела Михаила',
                                  201: 'Собор Успения Пресвятой Богородицы',
                                  202: 'Сормовский парк',
                                  203: 'Спасо-преображенская церковь',
                                  204: 'Спасо-преображенский собор',
                                  205: 'Спасский староярмарочный кафедральный собор',
                                  206: 'Старая аптека',
                                  207: 'Старинный велосипед',
                                  208: 'Старый Владимир',
                                  209: 'Суворовское училище',
                                  210: 'Тайницкая башня',
                                  211: 'Такса',
                                  212: 'Танкистам',
                                  213: 'Театр кукол',
                                  214: 'Театр музыкальной комедии',
                                  215: 'Театр оперы и балета',
                                  216: 'Театр фольклора «Разгуляй»',
                                  217: 'Торговый флигель О. Н. Каменевой',
                                  218: 'Троицкая церковь',
                                  219: 'Уральская государственная консерватория',
                                  220: 'Уральский геологический музей',
                                  221: 'Уральский государственный колледж имени И.И. Ползунова',
                                  222: 'Уральский государственный медицинский университет',
                                  223: 'Уральский государственный технический университет',
                                  224: 'Уральский государственный университет имени А. М. Горького',
                                  225: 'Уральский политехнический колледж',
                                  226: 'Уральский турбинный завод',
                                  227: 'Усадьба Вяхиревых',
                                  228: 'Усадьба Добролюбовых',
                                  229: 'Усадьба Е. И. Богоявленской',
                                  230: 'Усадьба Железнова',
                                  231: 'Усадьба Киршбаумов',
                                  232: 'Усадьба Расторгуевых — Харитоновых',
                                  233: 'Усадьба Рукавишниковых',
                                  234: 'Усадьба Тарасова',
                                  235: 'Усадьба ротмистра Переяславцева',
                                  236: 'Успенский Княгинин монастырь',
                                  237: 'Успенский собор на ВИЗе',
                                  238: 'Фавор',
                                  239: 'Филармония',
                                  240: 'Филёр',
                                  241: 'Харитоновский сад',
                                  242: 'Храм Живоначальной Троицы',
                                  243: 'Храм Казанской иконы Божией Матери',
                                  244: 'Храм Сергия Радонежского',
                                  245: 'Храм в честь Вознесения Господня',
                                  246: 'Храм на Крови',
                                  247: 'Храм святителей Московских',
                                  248: 'Храм святого Феофана Затворника при Архиерейском кабинете',
                                  249: 'Христианская Евангельская Церковь "Часовня у Голгофы"',
                                  250: 'Художественный Музей Эрнста Неизвестного',
                                  251: 'Художник на пленэре',
                                  252: 'Царский мост',
                                  253: 'Центр классической музыки',
                                  254: 'Центральная городская библиотека им. А. И. Герцена',
                                  255: 'Центральный стадион',
                                  256: 'Церковь Александра Невского в Сормово',
                                  257: 'Церковь Александра Невского при мужской гимназии',
                                  258: 'Церковь Воскресения Христова',
                                  259: 'Церковь Знамения Божией Матери и святых Жен-Мироносиц',
                                  260: 'Церковь Никиты Великомученика',
                                  261: 'Церковь Рождества Иоана Предтечи',
                                  262: 'Церковь Рождества Христова',
                                  263: 'Церковь в честь Преображения Господня',
                                  264: 'Церковь в честь Рождества Богородицы',
                                  265: 'Церковь во имя Всемилостивейшего Спаса',
                                  266: 'Церковь во имя Всемирлостивейшего Спаса',
                                  267: 'Церковь во имя Святого Пророка Божия Илии',
                                  268: 'Часовая башня',
                                  269: 'Часовня святого Николая Чудотворца, архиепископа Мирликийского',
                                  270: 'Чернобыль - трагедия XX века',
                                  271: 'Чкаловская лестница',
                                  272: 'Чкаловская лестница ',
                                  273: 'Шарташские каменные палатки',
                                  274: 'Шахматный дом (Игорный дом А.И. Троицкого)',
                                  275: 'Экспозиция «Подвиг народного единства»',
                                  276: 'улица Добролюбова',
                                  277: 'улица Февральской Революции',
                                  278: 'улица Чернышевского',
                                  279: '№12 Дом Севастьянова',
                                  280: '№15 Храм-на-Крови',
                                  281: '№16 Усадьба Расторгуева-Харитонова',
                                  282: '№17 Свердловская государственная академическая филармония',
                                  283: '№18 Фотографический музей «Дом Метенкова»',
                                  284: '№19 Музей истории Екатеринбурга',
                                  285: '№27 Дом Г.Н. Скрябина',
                                  286: '№28 Здание городской электростанции «Луч»',
                                  287: '№29 Дом В.И. Чувильдина',
                                  288: '№32 Дом обороны',
                                  289: '№34 Дом контор'}
        self.forward_transform = {v: k for k, v in self.inverse_transform.items()}
        self.city_map = {0: ['Нижний Новгород', 'Ярославль', 'Екатеринбург', 'Владимир'],
                         1: ['Нижний Новгород'],
                         2: ['Ярославль'], 3: ['Екатеринбург'], 4: ['Владимир']}

    def predict(self, img, city_id):
        city = self.city_map[city_id]

        image_data = base64.decodebytes(bytes(str(img), encoding='utf-8'))
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)

        prob = nn.Softmax(dim=1)(self.model(image)).squeeze().cpu()
        data = self.get_topk(prob, city)
        dist = self.get_dist(data)
        gen = []#self.generate(data.iloc[data['score'].argmax()])
        return {'result': {'categories': [{'label': label, 'prob': prob} for label, prob in dist.items()],
                           'objects': json_from_pandas_to_main_format(
                               data.to_json(orient='records', force_ascii=False))},
                'route': gen}

    def get_dist(self, data):
        dist = {'жилье': 0, 'археологические музеи': 0, 'археология': 0, 'архитектура': 0, 'художественные галереи': 0,
                'банк': 0, 'банки': 0, 'биографические музеи': 0, 'мосты': 0, 'места захоронений': 0, 'замки': 0,
                'соборы': 0, 'католические церкви': 0, 'кладбища': 0, 'детские музеи': 0, 'детские театры': 0,
                'церкви': 0, 'кинотеатры': 0, 'цирки': 0, 'концертные залы': 0, 'культурный': 0, 'защитные стены': 0,
                'разрушенные объекты': 0, 'восточные ортодоксальные церкви': 0, 'музеи моды': 0, 'продукты питания': 0,
                'фортификационные сооружения': 0, 'укрепленные башни': 0, 'фонтаны': 0, 'сады и парки': 0,
                'геологические образования': 0, 'исторические': 0, 'историческая архитектура': 0,
                'исторические районы': 0, 'исторические дома музеи': 0, 'исторические объекты': 0,
                'исторические поселения': 0, 'исторические места': 0, 'исторические музеи': 0,
                'промышленные объекты': 0, 'инсталляции': 0, 'интересные места': 0, 'кремли': 0, 'местные музеи': 0,
                'усадьбы': 0, 'военные музеи': 0, 'монастыри': 0, 'памятники': 0, 'монументы и памятники': 0,
                'мечети': 0, 'горные вершины': 0, 'музеи': 0, 'музеи науки и технологии': 0, 'музыкальные места': 0,
                'национальные музеи': 0, 'природные': 0, 'природные монументы': 0, 'природные заповедники': 0,
                'музеи под открытым небом': 0, 'оперные театры': 0, 'другие': 0, 'другие археологические объекты': 0,
                'другие мосты': 0, 'другие здания и сооружения': 0, 'другие места захоронения': 0, 'другие церкви': 0,
                'другие отели': 0, 'другие музеи': 0, 'другие природоохранные зоны': 0,
                'другие технологические музеи': 0, 'другие храмы': 0, 'другие театры': 0, 'другие башни': 0,
                'планетарии': 0, 'кукольные театры': 0, 'железнодорожные станции': 0, 'религия': 0, 'рестораны': 0,
                'скальные образования': 0, 'научные музеи': 0, 'скульптуры': 0, 'спорт': 0, 'площади': 0, 'стадионы': 0,
                'театры и развлечения': 0, 'туристические объекты': 0, 'туристический объект': 0, 'башни': 0,
                'триумфальные арки': 0, 'неклассифицированные объекты': 0, 'городская среда': 0,
                'смотровые площадки': 0, 'военные памятные места': 0, 'водяные башни': 0, 'зоопарки': 0, 'школы': 0,
                'образование': 0, 'университеты': 0}
        i = 0
        for _, line in data.iterrows():
            for kind in line['Kind']:
                dist[kind] += 1
                i += 1
        if i != 0:
            dist = {k: v / i for k, v in dist.items()}
        return dist

    def generate(self, root):
        self.train_data['Manhattan_distance'] = abs(self.train_data['Lon'] - root['Lon']) + abs(
            self.train_data['Lat'] - root['Lat'])
        sorted_df = self.train_data.sort_values(by='Manhattan_distance')

        nearest_places = sorted_df.iloc[:15]

        return nearest_places[['Lon', 'Lat']].values.tolist()

    def get_topk(self, pred, city, k=15):
        v, i = torch.topk(pred, k)
        vi = {i: v for i, v in zip(i.detach().numpy(), v.detach().numpy())}
        city_names = [self.inverse_transform[ind] for ind in i.numpy()]
        data = self.train_data.loc[self.train_data.Name.isin(city_names) & (self.train_data.City.isin(city))]
        scores = data['Name'].apply(lambda x: vi[self.forward_transform[x]])
        data['score'] = scores
        return data


