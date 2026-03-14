const CARD_TAG = "ha-washdata-card";
const EDITOR_TAG = "ha-washdata-card-editor";

const TRANSLATIONS = {
  "en": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "af": {
    "washer_program": "Wasprogram",
    "program_placeholder": "Kies Program",
    "duration": "Duur",
    "minutes": "min",
    "time_remaining": "Tyd wat oorbly",
    "no_prediction": "Geen voorspelling",
    "cycle_in_progress": "Siklus aan die gang",
    "status": "Status",
    "progress": "Vordering",
    "select_program": "Kies 'n program om besonderhede te sien",
    "title": "Titel",
    "status_entity": "Status Entiteit",
    "icon": "Ikoon",
    "active_color": "Aktiewe ikoonkleur",
    "show_state": "Wys Staat",
    "show_program": "Wys Program",
    "show_details": "Wys besonderhede",
    "spin_icon": "Draai-ikoon (Terwyl hardloop)",
    "program_entity": "Program Entiteit",
    "pct_entity": "Vorderingsentiteit (opsioneel)",
    "time_entity": "Tydsentiteit (opsioneel)",
    "display_mode": "Vertoonmodus",
    "show_time_remaining": "Wys oorblywende tyd",
    "show_percentage": "Wys persentasie",
    "entity_not_found": "Entiteit nie gevind nie"
  },
  "ar": {
    "washer_program": "برنامج الغساله",
    "program_placeholder": "حدد البرنامج",
    "duration": "مدة",
    "minutes": "دقيقة",
    "time_remaining": "الوقت المتبقي",
    "no_prediction": "لا التنبؤ",
    "cycle_in_progress": "الدورة قيد التقدم",
    "status": "حالة",
    "progress": "تقدم",
    "select_program": "حدد برنامجًا لمعرفة التفاصيل",
    "title": "عنوان",
    "status_entity": "كيان الحالة",
    "icon": "رمز",
    "active_color": "لون الرمز النشط",
    "show_state": "عرض الدولة",
    "show_program": "عرض البرنامج",
    "show_details": "إظهار التفاصيل",
    "spin_icon": "أيقونة الدوران (أثناء التشغيل)",
    "program_entity": "كيان البرنامج",
    "pct_entity": "كيان التقدم (اختياري)",
    "time_entity": "الكيان الزمني (اختياري)",
    "display_mode": "وضع العرض",
    "show_time_remaining": "عرض الوقت المتبقي",
    "show_percentage": "إظهار النسبة المئوية",
    "entity_not_found": "لم يتم العثور على الكيان"
  },
  "bg": {
    "washer_program": "Програма за пране",
    "program_placeholder": "Изберете Програма",
    "duration": "Продължителност",
    "minutes": "мин",
    "time_remaining": "Оставащо време",
    "no_prediction": "Няма прогноза",
    "cycle_in_progress": "Цикълът е в ход",
    "status": "Статус",
    "progress": "Напредък",
    "select_program": "Изберете програма, за да видите подробности",
    "title": "Заглавие",
    "status_entity": "Състояние на обекта",
    "icon": "Икона",
    "active_color": "Цвят на активната икона",
    "show_state": "Показване на състояние",
    "show_program": "Шоу програма",
    "show_details": "Показване на подробности",
    "spin_icon": "Въртяща се икона (докато работи)",
    "program_entity": "Програмен субект",
    "pct_entity": "Обект на напредъка (по избор)",
    "time_entity": "Времеви обект (по избор)",
    "display_mode": "Режим на показване",
    "show_time_remaining": "Показване на оставащото време",
    "show_percentage": "Показване на процента",
    "entity_not_found": "Обектът не е намерен"
  },
  "bn": {
    "washer_program": "ওয়াশার প্রোগ্রাম",
    "program_placeholder": "প্রোগ্রাম নির্বাচন করুন",
    "duration": "সময়কাল",
    "minutes": "মিনিট",
    "time_remaining": "বাকি সময়",
    "no_prediction": "কোন ভবিষ্যদ্বাণী নেই",
    "cycle_in_progress": "সাইকেল চলছে",
    "status": "স্ট্যাটাস",
    "progress": "অগ্রগতি",
    "select_program": "বিস্তারিত দেখতে একটি প্রোগ্রাম নির্বাচন করুন",
    "title": "শিরোনাম",
    "status_entity": "স্থিতি সত্তা",
    "icon": "আইকন",
    "active_color": "সক্রিয় আইকন রঙ",
    "show_state": "রাজ্য দেখান",
    "show_program": "প্রোগ্রাম দেখান",
    "show_details": "বিস্তারিত দেখান",
    "spin_icon": "স্পিনিং আইকন (চালানোর সময়)",
    "program_entity": "প্রোগ্রাম সত্তা",
    "pct_entity": "অগ্রগতি সত্তা (ঐচ্ছিক)",
    "time_entity": "সময়ের সত্তা (ঐচ্ছিক)",
    "display_mode": "প্রদর্শন মোড",
    "show_time_remaining": "অবশিষ্ট সময় দেখান",
    "show_percentage": "শতাংশ দেখান",
    "entity_not_found": "সত্তা খুঁজে পাওয়া যায়নি"
  },
  "bs": {
    "washer_program": "Program za pranje",
    "program_placeholder": "Odaberite Program",
    "duration": "Trajanje",
    "minutes": "min",
    "time_remaining": "Preostalo vrijeme",
    "no_prediction": "Nema predviđanja",
    "cycle_in_progress": "Ciklus je u toku",
    "status": "Status",
    "progress": "Napredak",
    "select_program": "Odaberite program da vidite detalje",
    "title": "Naslov",
    "status_entity": "Status Entiteta",
    "icon": "Ikona",
    "active_color": "Aktivna boja ikone",
    "show_state": "Prikaži državu",
    "show_program": "Show Program",
    "show_details": "Prikaži detalje",
    "spin_icon": "Ikona za okretanje (dok trčanje)",
    "program_entity": "Programski entitet",
    "pct_entity": "Entitet napretka (opciono)",
    "time_entity": "Vremenski entitet (opcionalno)",
    "display_mode": "Način prikaza",
    "show_time_remaining": "Prikaži preostalo vrijeme",
    "show_percentage": "Prikaži procenat",
    "entity_not_found": "Entitet nije pronađen"
  },
  "ca": {
    "washer_program": "Programa de rentadora",
    "program_placeholder": "Seleccioneu Programa",
    "duration": "Durada",
    "minutes": "min",
    "time_remaining": "Temps restant",
    "no_prediction": "Sense predicció",
    "cycle_in_progress": "Cicle en curs",
    "status": "Estat",
    "progress": "Progrés",
    "select_program": "Seleccioneu un programa per veure'n els detalls",
    "title": "Títol",
    "status_entity": "Entitat d'estat",
    "icon": "Icona",
    "active_color": "Color de la icona activa",
    "show_state": "Mostra l'estat",
    "show_program": "Programa Mostra",
    "show_details": "Mostra els detalls",
    "spin_icon": "Icona de gir (mentre corre)",
    "program_entity": "Entitat del programa",
    "pct_entity": "Entitat de progrés (opcional)",
    "time_entity": "Entitat temporal (opcional)",
    "display_mode": "Mode de visualització",
    "show_time_remaining": "Mostra el temps restant",
    "show_percentage": "Mostra el percentatge",
    "entity_not_found": "No s'ha trobat l'entitat"
  },
  "cs": {
    "washer_program": "Program pračky",
    "program_placeholder": "Vyberte Program",
    "duration": "Trvání",
    "minutes": "min",
    "time_remaining": "Zbývající čas",
    "no_prediction": "Žádná předpověď",
    "cycle_in_progress": "Cyklus probíhá",
    "status": "Postavení",
    "progress": "Pokrok",
    "select_program": "Chcete-li zobrazit podrobnosti, vyberte program",
    "title": "Titul",
    "status_entity": "Stavová entita",
    "icon": "Ikona",
    "active_color": "Barva aktivní ikony",
    "show_state": "Zobrazit stav",
    "show_program": "Zobrazit program",
    "show_details": "Zobrazit podrobnosti",
    "spin_icon": "Ikona rotace (při běhu)",
    "program_entity": "Entita programu",
    "pct_entity": "Entita průběhu (volitelné)",
    "time_entity": "Časová entita (volitelné)",
    "display_mode": "Režim zobrazení",
    "show_time_remaining": "Zobrazit zbývající čas",
    "show_percentage": "Zobrazit procento",
    "entity_not_found": "Entita nenalezena"
  },
  "cy": {
    "washer_program": "Rhaglen Wasier",
    "program_placeholder": "Dewiswch Rhaglen",
    "duration": "Hyd",
    "minutes": "min",
    "time_remaining": "Amser yn weddill",
    "no_prediction": "Dim Rhagfynegiad",
    "cycle_in_progress": "Cylch ar y gweill",
    "status": "Statws",
    "progress": "Cynnydd",
    "select_program": "Dewiswch raglen i weld y manylion",
    "title": "Teitl",
    "status_entity": "Endid Statws",
    "icon": "Eicon",
    "active_color": "Lliw Eicon Actif",
    "show_state": "Dangos Cyflwr",
    "show_program": "Rhaglen Sioe",
    "show_details": "Dangos Manylion",
    "spin_icon": "Eicon Troelli (Wrth redeg)",
    "program_entity": "Endid Rhaglen",
    "pct_entity": "Endid Cynnydd (Dewisol)",
    "time_entity": "Endid Amser (Dewisol)",
    "display_mode": "Modd Arddangos",
    "show_time_remaining": "Dangos Amser ar ôl",
    "show_percentage": "Dangos Canran",
    "entity_not_found": "Endid heb ei ganfod"
  },
  "da": {
    "washer_program": "Vaskeprogram",
    "program_placeholder": "Vælg Program",
    "duration": "Varighed",
    "minutes": "min",
    "time_remaining": "Tid tilbage",
    "no_prediction": "Ingen forudsigelse",
    "cycle_in_progress": "Cyklus i gang",
    "status": "Status",
    "progress": "Fremskridt",
    "select_program": "Vælg et program for at se detaljer",
    "title": "Titel",
    "status_entity": "Statusenhed",
    "icon": "Ikon",
    "active_color": "Aktiv ikon farve",
    "show_state": "Vis tilstand",
    "show_program": "Vis program",
    "show_details": "Vis detaljer",
    "spin_icon": "Spinning-ikon (mens du løber)",
    "program_entity": "Programenhed",
    "pct_entity": "Fremskridtsenhed (valgfrit)",
    "time_entity": "Tidsenhed (valgfrit)",
    "display_mode": "Visningstilstand",
    "show_time_remaining": "Vis resterende tid",
    "show_percentage": "Vis procent",
    "entity_not_found": "Enheden blev ikke fundet"
  },
  "de": {
    "washer_program": "Waschprogramm",
    "program_placeholder": "Wählen Sie Programm",
    "duration": "Dauer",
    "minutes": "min",
    "time_remaining": "Verbleibende Zeit",
    "no_prediction": "Keine Vorhersage",
    "cycle_in_progress": "Zyklus läuft",
    "status": "Status",
    "progress": "Fortschritt",
    "select_program": "Wählen Sie ein Programm aus, um Details anzuzeigen",
    "title": "Titel",
    "status_entity": "Status-Entität",
    "icon": "Symbol",
    "active_color": "Aktive Symbolfarbe",
    "show_state": "Status anzeigen",
    "show_program": "Programm anzeigen",
    "show_details": "Details anzeigen",
    "spin_icon": "Spinning-Symbol (während des Laufens)",
    "program_entity": "Programmeinheit",
    "pct_entity": "Fortschrittsentität (optional)",
    "time_entity": "Zeiteinheit (optional)",
    "display_mode": "Anzeigemodus",
    "show_time_remaining": "Verbleibende Zeit anzeigen",
    "show_percentage": "Prozentsatz anzeigen",
    "entity_not_found": "Entität nicht gefunden"
  },
  "el": {
    "washer_program": "Πρόγραμμα πλύσης",
    "program_placeholder": "Επιλέξτε Πρόγραμμα",
    "duration": "Διάρκεια",
    "minutes": "ελάχ",
    "time_remaining": "Χρόνος που απομένει",
    "no_prediction": "Καμία Πρόβλεψη",
    "cycle_in_progress": "Κύκλος σε εξέλιξη",
    "status": "Κατάσταση",
    "progress": "Πρόοδος",
    "select_program": "Επιλέξτε ένα πρόγραμμα για να δείτε λεπτομέρειες",
    "title": "Τίτλος",
    "status_entity": "Οντότητα κατάστασης",
    "icon": "Εικόνισμα",
    "active_color": "Χρώμα ενεργού εικονιδίου",
    "show_state": "Εμφάνιση κατάστασης",
    "show_program": "Εμφάνιση προγράμματος",
    "show_details": "Εμφάνιση λεπτομερειών",
    "spin_icon": "Περιστρεφόμενο εικονίδιο (Κατά την εκτέλεση)",
    "program_entity": "Οντότητα προγράμματος",
    "pct_entity": "Οντότητα προόδου (Προαιρετικό)",
    "time_entity": "Οντότητα ώρας (Προαιρετικό)",
    "display_mode": "Λειτουργία εμφάνισης",
    "show_time_remaining": "Εμφάνιση χρόνου που απομένει",
    "show_percentage": "Εμφάνιση ποσοστού",
    "entity_not_found": "Η οντότητα δεν βρέθηκε"
  },
  "en-GB": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "eo": {
    "washer_program": "Programo de Lavujo",
    "program_placeholder": "Elektu Programon",
    "duration": "Daŭro",
    "minutes": "min",
    "time_remaining": "Tempo Restanta",
    "no_prediction": "Neniu Antaŭdiro",
    "cycle_in_progress": "Ciklo en progreso",
    "status": "Statuso",
    "progress": "Progreso",
    "select_program": "Elektu programon por vidi detalojn",
    "title": "Titolo",
    "status_entity": "Statusa Ento",
    "icon": "Ikono",
    "active_color": "Aktiva Ikono Koloro",
    "show_state": "Montru Ŝtaton",
    "show_program": "Montru Programon",
    "show_details": "Montru Detalojn",
    "spin_icon": "Turniĝanta Ikono (Dum kurado)",
    "program_entity": "Programa Ento",
    "pct_entity": "Progresa Ento (Laŭvola)",
    "time_entity": "Tempo-Entaĵo (Laŭvola)",
    "display_mode": "Montra Reĝimo",
    "show_time_remaining": "Montru Restantan Tempon",
    "show_percentage": "Montru Procenton",
    "entity_not_found": "Ento ne trovita"
  },
  "es": {
    "washer_program": "Programa de lavadora",
    "program_placeholder": "Seleccionar programa",
    "duration": "Duración",
    "minutes": "mín.",
    "time_remaining": "Tiempo restante",
    "no_prediction": "Sin predicción",
    "cycle_in_progress": "Ciclo en progreso",
    "status": "Estado",
    "progress": "Progreso",
    "select_program": "Selecciona un programa para ver detalles",
    "title": "Título",
    "status_entity": "Entidad de estado",
    "icon": "Icono",
    "active_color": "Color del icono activo",
    "show_state": "Mostrar estado",
    "show_program": "Mostrar programa",
    "show_details": "Mostrar detalles",
    "spin_icon": "Icono de giro (mientras se ejecuta)",
    "program_entity": "Entidad del programa",
    "pct_entity": "Entidad de progreso (opcional)",
    "time_entity": "Entidad de tiempo (opcional)",
    "display_mode": "Modo de visualización",
    "show_time_remaining": "Mostrar tiempo restante",
    "show_percentage": "Mostrar porcentaje",
    "entity_not_found": "Entidad no encontrada"
  },
  "es-419": {
    "washer_program": "Programa de lavadora",
    "program_placeholder": "Seleccionar programa",
    "duration": "Duración",
    "minutes": "mín.",
    "time_remaining": "Tiempo restante",
    "no_prediction": "Sin predicción",
    "cycle_in_progress": "Ciclo en progreso",
    "status": "Estado",
    "progress": "Progreso",
    "select_program": "Selecciona un programa para ver detalles",
    "title": "Título",
    "status_entity": "Entidad de estado",
    "icon": "Icono",
    "active_color": "Color del icono activo",
    "show_state": "Mostrar estado",
    "show_program": "Mostrar programa",
    "show_details": "Mostrar detalles",
    "spin_icon": "Icono de giro (mientras se ejecuta)",
    "program_entity": "Entidad del programa",
    "pct_entity": "Entidad de progreso (opcional)",
    "time_entity": "Entidad de tiempo (opcional)",
    "display_mode": "Modo de visualización",
    "show_time_remaining": "Mostrar tiempo restante",
    "show_percentage": "Mostrar porcentaje",
    "entity_not_found": "Entidad no encontrada"
  },
  "et": {
    "washer_program": "Pesumasina programm",
    "program_placeholder": "Valige Programm",
    "duration": "Kestus",
    "minutes": "min",
    "time_remaining": "Järelejäänud aeg",
    "no_prediction": "Ei mingit ennustust",
    "cycle_in_progress": "Tsükkel on pooleli",
    "status": "Olek",
    "progress": "Edusammud",
    "select_program": "Üksikasjade vaatamiseks valige programm",
    "title": "Pealkiri",
    "status_entity": "Olekuüksus",
    "icon": "Ikoon",
    "active_color": "Aktiivne ikooni värv",
    "show_state": "Näita olekut",
    "show_program": "Näita programmi",
    "show_details": "Näita üksikasju",
    "spin_icon": "Pöörlev ikoon (jooksmise ajal)",
    "program_entity": "Programmi üksus",
    "pct_entity": "Edenemisüksus (valikuline)",
    "time_entity": "Ajaüksus (valikuline)",
    "display_mode": "Kuvamisrežiim",
    "show_time_remaining": "Näita järelejäänud aega",
    "show_percentage": "Näita protsenti",
    "entity_not_found": "Üksust ei leitud"
  },
  "eu": {
    "washer_program": "Garbigailuen programa",
    "program_placeholder": "Hautatu Programa",
    "duration": "Iraupena",
    "minutes": "min",
    "time_remaining": "Gelditzen den denbora",
    "no_prediction": "Iragarpenik ez",
    "cycle_in_progress": "Zikloa martxan",
    "status": "Egoera",
    "progress": "Aurrerapena",
    "select_program": "Hautatu programa bat xehetasunak ikusteko",
    "title": "Izenburua",
    "status_entity": "Egoera Entitatea",
    "icon": "Ikonoa",
    "active_color": "Ikono aktiboaren kolorea",
    "show_state": "Erakutsi egoera",
    "show_program": "Erakutsi programa",
    "show_details": "Erakutsi xehetasunak",
    "spin_icon": "Biratzen ari den ikonoa (exekutatzen ari zaren bitartean)",
    "program_entity": "Programa Entitatea",
    "pct_entity": "Aurrerapen-entitatea (aukerakoa)",
    "time_entity": "Denbora-entitatea (aukerakoa)",
    "display_mode": "Bistaratzeko modua",
    "show_time_remaining": "Erakutsi falta den denbora",
    "show_percentage": "Erakutsi ehunekoa",
    "entity_not_found": "Ez da aurkitu entitatea"
  },
  "fa": {
    "washer_program": "برنامه شستشو",
    "program_placeholder": "برنامه را انتخاب کنید",
    "duration": "مدت زمان",
    "minutes": "دقیقه",
    "time_remaining": "زمان باقی مانده",
    "no_prediction": "بدون پیش بینی",
    "cycle_in_progress": "چرخه در حال انجام است",
    "status": "وضعیت",
    "progress": "پیشرفت",
    "select_program": "یک برنامه را برای دیدن جزئیات انتخاب کنید",
    "title": "عنوان",
    "status_entity": "موجودیت وضعیت",
    "icon": "نماد",
    "active_color": "رنگ آیکون فعال",
    "show_state": "نمایش وضعیت",
    "show_program": "نمایش برنامه",
    "show_details": "نمایش جزئیات",
    "spin_icon": "نماد چرخان (هنگام اجرا)",
    "program_entity": "نهاد برنامه",
    "pct_entity": "موجودیت پیشرفت (اختیاری)",
    "time_entity": "موجودیت زمان (اختیاری)",
    "display_mode": "حالت نمایش",
    "show_time_remaining": "نمایش زمان باقی مانده",
    "show_percentage": "نمایش درصد",
    "entity_not_found": "موجودیت یافت نشد"
  },
  "fi": {
    "washer_program": "Pesuohjelma",
    "program_placeholder": "Valitse Ohjelma",
    "duration": "Kesto",
    "minutes": "min",
    "time_remaining": "Aikaa jäljellä",
    "no_prediction": "Ei ennustetta",
    "cycle_in_progress": "Kierto käynnissä",
    "status": "Status",
    "progress": "Edistyminen",
    "select_program": "Valitse ohjelma nähdäksesi tiedot",
    "title": "Otsikko",
    "status_entity": "Status Entity",
    "icon": "Kuvake",
    "active_color": "Aktiivinen kuvakkeen väri",
    "show_state": "Näytä tila",
    "show_program": "Näytä ohjelma",
    "show_details": "Näytä tiedot",
    "spin_icon": "Pyörivä kuvake (juoksessa)",
    "program_entity": "Ohjelmakokonaisuus",
    "pct_entity": "Etenemiskokonaisuus (valinnainen)",
    "time_entity": "Aikakokonaisuus (valinnainen)",
    "display_mode": "Näyttötila",
    "show_time_remaining": "Näytä jäljellä oleva aika",
    "show_percentage": "Näytä prosenttiosuus",
    "entity_not_found": "Kokonaisuutta ei löydy"
  },
  "fr": {
    "washer_program": "Programme de laveuse",
    "program_placeholder": "Sélectionnez le programme",
    "duration": "Durée",
    "minutes": "min",
    "time_remaining": "Temps restant",
    "no_prediction": "Aucune prédiction",
    "cycle_in_progress": "Cycle en cours",
    "status": "Statut",
    "progress": "Progrès",
    "select_program": "Sélectionnez un programme pour voir les détails",
    "title": "Titre",
    "status_entity": "Entité de statut",
    "icon": "Icône",
    "active_color": "Couleur de l'icône active",
    "show_state": "Afficher l'état",
    "show_program": "Programme du spectacle",
    "show_details": "Afficher les détails",
    "spin_icon": "Icône de rotation (pendant l'exécution)",
    "program_entity": "Entité du programme",
    "pct_entity": "Entité de progression (facultatif)",
    "time_entity": "Entité temporelle (facultatif)",
    "display_mode": "Mode d'affichage",
    "show_time_remaining": "Afficher le temps restant",
    "show_percentage": "Afficher le pourcentage",
    "entity_not_found": "Entité introuvable"
  },
  "fy": {
    "washer_program": "Washer Program",
    "program_placeholder": "Selektearje Programma",
    "duration": "Doer",
    "minutes": "min",
    "time_remaining": "Tiid oerbleaun",
    "no_prediction": "Gjin foarsizzing",
    "cycle_in_progress": "Cycle yn útfiering",
    "status": "Status",
    "progress": "Foarútgong",
    "select_program": "Selektearje in programma om details te sjen",
    "title": "Titel",
    "status_entity": "Status Entiteit",
    "icon": "Ikoan",
    "active_color": "Aktive ikoankleur",
    "show_state": "Steat sjen litte",
    "show_program": "Programma sjen litte",
    "show_details": "Show Details",
    "spin_icon": "Spinnend ikoan (by it rinnen)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (opsjoneel)",
    "time_entity": "Tiid entiteit (opsjoneel)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Lit de oerbleaune tiid sjen",
    "show_percentage": "Persintaazje sjen litte",
    "entity_not_found": "Entiteit net fûn"
  },
  "ga": {
    "washer_program": "Clár níocháin",
    "program_placeholder": "Roghnaigh Clár",
    "duration": "Fad",
    "minutes": "nóim",
    "time_remaining": "Am fágtha",
    "no_prediction": "Gan Tuar",
    "cycle_in_progress": "Timthriall ar siúl",
    "status": "Stádas",
    "progress": "Dul chun cinn",
    "select_program": "Roghnaigh clár chun sonraí a fheiceáil",
    "title": "Teideal",
    "status_entity": "Aonán Stádais",
    "icon": "Deilbhín",
    "active_color": "Dath Deilbhín Gníomhach",
    "show_state": "Taispeáin Stáit",
    "show_program": "Clár Taispeáin",
    "show_details": "Taispeáin Sonraí",
    "spin_icon": "Deilbhín Casadh (Agus tú ag rith)",
    "program_entity": "Aonán Cláir",
    "pct_entity": "Aonán Dul Chun Cinn (Roghnach)",
    "time_entity": "Aonán Ama (Roghnach)",
    "display_mode": "Mód Taispeána",
    "show_time_remaining": "Taispeáin Am fágtha",
    "show_percentage": "Taispeáin Céatadán",
    "entity_not_found": "Aonán gan aimsiú"
  },
  "gl": {
    "washer_program": "Programa Lavadora",
    "program_placeholder": "Seleccione Programa",
    "duration": "Duración",
    "minutes": "min",
    "time_remaining": "Tempo Restante",
    "no_prediction": "Sen predición",
    "cycle_in_progress": "Ciclo en curso",
    "status": "Estado",
    "progress": "Progreso",
    "select_program": "Seleccione un programa para ver os detalles",
    "title": "Título",
    "status_entity": "Entidade de estado",
    "icon": "Icona",
    "active_color": "Cor da icona activa",
    "show_state": "Mostrar estado",
    "show_program": "Programa Mostrar",
    "show_details": "Mostrar detalles",
    "spin_icon": "Icona xirando (mentres corres)",
    "program_entity": "Entidade do programa",
    "pct_entity": "Entidade de progreso (opcional)",
    "time_entity": "Entidade horaria (opcional)",
    "display_mode": "Modo de visualización",
    "show_time_remaining": "Mostrar o tempo restante",
    "show_percentage": "Mostrar porcentaxe",
    "entity_not_found": "Non se atopou a entidade"
  },
  "gsw": {
    "washer_program": "Waschprogramm",
    "program_placeholder": "Wählen Sie Programm",
    "duration": "Dauer",
    "minutes": "min",
    "time_remaining": "Verbleibende Zeit",
    "no_prediction": "Keine Vorhersage",
    "cycle_in_progress": "Zyklus läuft",
    "status": "Status",
    "progress": "Fortschritt",
    "select_program": "Wählen Sie ein Programm aus, um Details anzuzeigen",
    "title": "Titel",
    "status_entity": "Status-Entität",
    "icon": "Symbol",
    "active_color": "Aktive Symbolfarbe",
    "show_state": "Status anzeigen",
    "show_program": "Programm anzeigen",
    "show_details": "Details anzeigen",
    "spin_icon": "Spinning-Symbol (während des Laufens)",
    "program_entity": "Programmeinheit",
    "pct_entity": "Fortschrittsentität (optional)",
    "time_entity": "Zeiteinheit (optional)",
    "display_mode": "Anzeigemodus",
    "show_time_remaining": "Verbleibende Zeit anzeigen",
    "show_percentage": "Prozentsatz anzeigen",
    "entity_not_found": "Entität nicht gefunden"
  },
  "he": {
    "washer_program": "תוכנית כביסה",
    "program_placeholder": "בחר תוכנית",
    "duration": "מֶשֶׁך",
    "minutes": "דקה",
    "time_remaining": "זמן שנותר",
    "no_prediction": "אין תחזית",
    "cycle_in_progress": "מחזור בעיצומו",
    "status": "סטָטוּס",
    "progress": "הִתקַדְמוּת",
    "select_program": "בחר תוכנית כדי לראות פרטים",
    "title": "כּוֹתֶרֶת",
    "status_entity": "ישות סטטוס",
    "icon": "סמל",
    "active_color": "צבע סמל פעיל",
    "show_state": "הצג מדינה",
    "show_program": "הצג תוכנית",
    "show_details": "הצג פרטים",
    "spin_icon": "סמל מסתובב (תוך כדי ריצה)",
    "program_entity": "ישות תוכנית",
    "pct_entity": "ישות התקדמות (אופציונלי)",
    "time_entity": "ישות זמן (אופציונלי)",
    "display_mode": "מצב תצוגה",
    "show_time_remaining": "הצג את הזמן שנותר",
    "show_percentage": "הצג אחוז",
    "entity_not_found": "הישות לא נמצאה"
  },
  "hi": {
    "washer_program": "वॉशर कार्यक्रम",
    "program_placeholder": "प्रोग्राम चुनें",
    "duration": "अवधि",
    "minutes": "मिन",
    "time_remaining": "शेष समय",
    "no_prediction": "कोई भविष्यवाणी नहीं",
    "cycle_in_progress": "चक्र चल रहा है",
    "status": "स्थिति",
    "progress": "प्रगति",
    "select_program": "विवरण देखने के लिए किसी प्रोग्राम का चयन करें",
    "title": "शीर्षक",
    "status_entity": "स्थिति इकाई",
    "icon": "आइकन",
    "active_color": "सक्रिय चिह्न रंग",
    "show_state": "राज्य दिखाएँ",
    "show_program": "कार्यक्रम दिखाएँ",
    "show_details": "प्रदर्शन का विवरण",
    "spin_icon": "घूमता हुआ चिह्न (दौड़ते समय)",
    "program_entity": "कार्यक्रम इकाई",
    "pct_entity": "प्रगति इकाई (वैकल्पिक)",
    "time_entity": "समय इकाई (वैकल्पिक)",
    "display_mode": "प्रदर्शन मोड",
    "show_time_remaining": "शेष समय दिखाएँ",
    "show_percentage": "प्रतिशत दिखाएँ",
    "entity_not_found": "इकाई नहीं मिली"
  },
  "hr": {
    "washer_program": "Program za pranje",
    "program_placeholder": "Odaberite Program",
    "duration": "Trajanje",
    "minutes": "min",
    "time_remaining": "Preostalo vrijeme",
    "no_prediction": "Nema predviđanja",
    "cycle_in_progress": "Ciklus u tijeku",
    "status": "Status",
    "progress": "Napredak",
    "select_program": "Odaberite program da biste vidjeli pojedinosti",
    "title": "Titula",
    "status_entity": "Statusni entitet",
    "icon": "Ikona",
    "active_color": "Boja aktivne ikone",
    "show_state": "Prikaži stanje",
    "show_program": "Show Program",
    "show_details": "Prikaži pojedinosti",
    "spin_icon": "Ikona koja se vrti (dok radi)",
    "program_entity": "Programski entitet",
    "pct_entity": "Entitet napretka (neobavezno)",
    "time_entity": "Entitet vremena (neobavezno)",
    "display_mode": "Način prikaza",
    "show_time_remaining": "Prikaži preostalo vrijeme",
    "show_percentage": "Prikaži postotak",
    "entity_not_found": "Entitet nije pronađen"
  },
  "hu": {
    "washer_program": "Mosó program",
    "program_placeholder": "Válassza a Program lehetőséget",
    "duration": "Időtartam",
    "minutes": "min",
    "time_remaining": "Hátralévő idő",
    "no_prediction": "Nincs előrejelzés",
    "cycle_in_progress": "Ciklus folyamatban",
    "status": "Állapot",
    "progress": "Előrehalad",
    "select_program": "Válasszon ki egy programot a részletek megtekintéséhez",
    "title": "Cím",
    "status_entity": "Állapot entitás",
    "icon": "Ikon",
    "active_color": "Aktív Ikon színe",
    "show_state": "Állapot megjelenítése",
    "show_program": "Program megjelenítése",
    "show_details": "Részletek megjelenítése",
    "spin_icon": "Pörgő ikon (futás közben)",
    "program_entity": "Program entitás",
    "pct_entity": "Haladási entitás (opcionális)",
    "time_entity": "Idő entitás (opcionális)",
    "display_mode": "Kijelző mód",
    "show_time_remaining": "Mutasd a hátralévő időt",
    "show_percentage": "Százalék megjelenítése",
    "entity_not_found": "Az entitás nem található"
  },
  "hy": {
    "washer_program": "Լվացքի ծրագիր",
    "program_placeholder": "Ընտրեք Ծրագիր",
    "duration": "Տևողությունը",
    "minutes": "ր",
    "time_remaining": "Մնացած ժամանակը",
    "no_prediction": "Ոչ մի կանխատեսում",
    "cycle_in_progress": "Ցիկլը ընթացքի մեջ է",
    "status": "Կարգավիճակ",
    "progress": "Առաջընթաց",
    "select_program": "Մանրամասները տեսնելու համար ընտրեք ծրագիր",
    "title": "Վերնագիր",
    "status_entity": "Կարգավիճակի սուբյեկտ",
    "icon": "Սրբապատկեր",
    "active_color": "Ակտիվ պատկերակի գույնը",
    "show_state": "Ցույց տալ վիճակը",
    "show_program": "Ցույց տալ ծրագիրը",
    "show_details": "Ցույց տալ մանրամասները",
    "spin_icon": "Պտտվող պատկերակ (վազքի ընթացքում)",
    "program_entity": "Ծրագրի սուբյեկտ",
    "pct_entity": "Առաջընթաց կազմակերպություն (ըստ ցանկության)",
    "time_entity": "Ժամանակի միավոր (ըստ ցանկության)",
    "display_mode": "Ցուցադրման ռեժիմ",
    "show_time_remaining": "Ցույց տալ Մնացած ժամանակը",
    "show_percentage": "Ցույց տալ տոկոսը",
    "entity_not_found": "Կազմակերպությունը չի գտնվել"
  },
  "id": {
    "washer_program": "Program Mesin Cuci",
    "program_placeholder": "Pilih Program",
    "duration": "Lamanya",
    "minutes": "menit",
    "time_remaining": "Sisa Waktu",
    "no_prediction": "Tidak Ada Prediksi",
    "cycle_in_progress": "Siklus sedang berlangsung",
    "status": "Status",
    "progress": "Kemajuan",
    "select_program": "Pilih program untuk melihat detailnya",
    "title": "Judul",
    "status_entity": "Entitas Status",
    "icon": "Ikon",
    "active_color": "Warna Ikon Aktif",
    "show_state": "Tampilkan Negara",
    "show_program": "Tampilkan Program",
    "show_details": "Tampilkan Detail",
    "spin_icon": "Ikon Berputar (Saat berlari)",
    "program_entity": "Entitas Program",
    "pct_entity": "Entitas Kemajuan (Opsional)",
    "time_entity": "Entitas Waktu (Opsional)",
    "display_mode": "Modus Tampilan",
    "show_time_remaining": "Tampilkan Sisa Waktu",
    "show_percentage": "Tampilkan Persentase",
    "entity_not_found": "Entitas tidak ditemukan"
  },
  "is": {
    "washer_program": "Þvottavélaforrit",
    "program_placeholder": "Veldu Program",
    "duration": "Lengd",
    "minutes": "mín",
    "time_remaining": "Tími sem eftir er",
    "no_prediction": "Engin spá",
    "cycle_in_progress": "Hringrás í gangi",
    "status": "Staða",
    "progress": "Framfarir",
    "select_program": "Veldu forrit til að sjá upplýsingar",
    "title": "Titill",
    "status_entity": "Staða eining",
    "icon": "Táknmynd",
    "active_color": "Virkur táknlitur",
    "show_state": "Sýna ástand",
    "show_program": "Sýna dagskrá",
    "show_details": "Sýna upplýsingar",
    "spin_icon": "Snúningstákn (meðan í gangi)",
    "program_entity": "Dagskráreining",
    "pct_entity": "Framvindueining (valfrjálst)",
    "time_entity": "Tímaeining (valfrjálst)",
    "display_mode": "Sýnastilling",
    "show_time_remaining": "Sýna tíma sem eftir er",
    "show_percentage": "Sýna hlutfall",
    "entity_not_found": "Eining fannst ekki"
  },
  "it": {
    "washer_program": "Programma Lavatrice",
    "program_placeholder": "Seleziona Programma",
    "duration": "Durata",
    "minutes": "min",
    "time_remaining": "Tempo rimanente",
    "no_prediction": "Nessuna previsione",
    "cycle_in_progress": "Ciclo in corso",
    "status": "Stato",
    "progress": "Progressi",
    "select_program": "Seleziona un programma per vedere i dettagli",
    "title": "Titolo",
    "status_entity": "Entità di stato",
    "icon": "Icona",
    "active_color": "Colore icona attiva",
    "show_state": "Mostra stato",
    "show_program": "Mostra programma",
    "show_details": "Mostra dettagli",
    "spin_icon": "Icona che gira (durante la corsa)",
    "program_entity": "Entità del programma",
    "pct_entity": "Entità di avanzamento (facoltativo)",
    "time_entity": "Entità temporale (facoltativo)",
    "display_mode": "Modalità di visualizzazione",
    "show_time_remaining": "Mostra tempo rimanente",
    "show_percentage": "Mostra percentuale",
    "entity_not_found": "Entità non trovata"
  },
  "ja": {
    "washer_program": "ウォッシャープログラム",
    "program_placeholder": "プログラムの選択",
    "duration": "間隔",
    "minutes": "分",
    "time_remaining": "残り時間",
    "no_prediction": "予測なし",
    "cycle_in_progress": "進行中のサイクル",
    "status": "状態",
    "progress": "進捗",
    "select_program": "プログラムを選択して詳細を表示します",
    "title": "タイトル",
    "status_entity": "ステータスエンティティ",
    "icon": "アイコン",
    "active_color": "アクティブなアイコンの色",
    "show_state": "状態を表示",
    "show_program": "ショープログラム",
    "show_details": "詳細を表示",
    "spin_icon": "回転アイコン（走行中）",
    "program_entity": "プログラムエンティティ",
    "pct_entity": "進行状況エンティティ (オプション)",
    "time_entity": "時間エンティティ (オプション)",
    "display_mode": "表示モード",
    "show_time_remaining": "残りの上映時間",
    "show_percentage": "パーセンテージを表示",
    "entity_not_found": "エンティティが見つかりません"
  },
  "ka": {
    "washer_program": "სარეცხი პროგრამა",
    "program_placeholder": "აირჩიეთ პროგრამა",
    "duration": "ხანგრძლივობა",
    "minutes": "წთ",
    "time_remaining": "დარჩენილი დრო",
    "no_prediction": "არანაირი პროგნოზი",
    "cycle_in_progress": "ციკლი მიმდინარეობს",
    "status": "სტატუსი",
    "progress": "პროგრესი",
    "select_program": "აირჩიეთ პროგრამა დეტალების სანახავად",
    "title": "სათაური",
    "status_entity": "სტატუსის ერთეული",
    "icon": "ხატულა",
    "active_color": "აქტიური ხატის ფერი",
    "show_state": "სახელმწიფოს ჩვენება",
    "show_program": "პროგრამის ჩვენება",
    "show_details": "დეტალების ჩვენება",
    "spin_icon": "დაწნული ხატულა (გაშვებისას)",
    "program_entity": "პროგრამის სუბიექტი",
    "pct_entity": "პროგრესული ერთეული (არასავალდებულო)",
    "time_entity": "დროის ერთეული (არასავალდებულო)",
    "display_mode": "ჩვენების რეჟიმი",
    "show_time_remaining": "დარჩენილი დროის ჩვენება",
    "show_percentage": "პროცენტის ჩვენება",
    "entity_not_found": "ერთეული ვერ მოიძებნა"
  },
  "ko": {
    "washer_program": "세탁기 프로그램",
    "program_placeholder": "프로그램 선택",
    "duration": "지속",
    "minutes": "분",
    "time_remaining": "남은 시간",
    "no_prediction": "예측 없음",
    "cycle_in_progress": "사이클 진행 중",
    "status": "상태",
    "progress": "진전",
    "select_program": "세부정보를 보려면 프로그램을 선택하세요.",
    "title": "제목",
    "status_entity": "상태 엔터티",
    "icon": "상",
    "active_color": "활성 아이콘 색상",
    "show_state": "상태 표시",
    "show_program": "쇼 프로그램",
    "show_details": "세부정보 표시",
    "spin_icon": "회전 아이콘(실행 중)",
    "program_entity": "프로그램 엔터티",
    "pct_entity": "진행 엔터티(선택 사항)",
    "time_entity": "시간 엔터티(선택 사항)",
    "display_mode": "디스플레이 모드",
    "show_time_remaining": "남은 시간 표시",
    "show_percentage": "백분율 표시",
    "entity_not_found": "엔터티를 찾을 수 없습니다."
  },
  "lb": {
    "washer_program": "Wäschmaschinn Programm",
    "program_placeholder": "Wielt Programm",
    "duration": "Dauer",
    "minutes": "min",
    "time_remaining": "Zäit Rescht",
    "no_prediction": "Keng Prognose",
    "cycle_in_progress": "Zyklus amgaang",
    "status": "Status",
    "progress": "Fortschrëtt",
    "select_program": "Wielt e Programm fir Detailer ze gesinn",
    "title": "Titel",
    "status_entity": "Status Entitéit",
    "icon": "Ikon",
    "active_color": "Aktiv Ikon Faarf",
    "show_state": "Staat weisen",
    "show_program": "Show Programm",
    "show_details": "Show Detailer",
    "spin_icon": "Spinning Ikon (Wärend Lafen)",
    "program_entity": "Programm Entitéit",
    "pct_entity": "Progress Entity (fakultativ)",
    "time_entity": "Zäit Entitéit (optional)",
    "display_mode": "Display Modus",
    "show_time_remaining": "Show Rescht Zäit",
    "show_percentage": "Show Prozentsaz",
    "entity_not_found": "Entitéit net fonnt"
  },
  "lt": {
    "washer_program": "Skalbimo programa",
    "program_placeholder": "Pasirinkite Programa",
    "duration": "Trukmė",
    "minutes": "min",
    "time_remaining": "Likęs laikas",
    "no_prediction": "Jokios prognozės",
    "cycle_in_progress": "Vyksta ciklas",
    "status": "Būsena",
    "progress": "Pažanga",
    "select_program": "Norėdami pamatyti išsamią informaciją, pasirinkite programą",
    "title": "Pavadinimas",
    "status_entity": "Būsenos subjektas",
    "icon": "Piktograma",
    "active_color": "Aktyvios piktogramos spalva",
    "show_state": "Rodyti būseną",
    "show_program": "Rodyti programą",
    "show_details": "Rodyti išsamią informaciją",
    "spin_icon": "Sukimo piktograma (bėgant)",
    "program_entity": "Programos subjektas",
    "pct_entity": "Pažangos subjektas (neprivaloma)",
    "time_entity": "Laiko objektas (neprivaloma)",
    "display_mode": "Ekrano režimas",
    "show_time_remaining": "Rodyti likusį laiką",
    "show_percentage": "Rodyti procentą",
    "entity_not_found": "Subjektas nerastas"
  },
  "lv": {
    "washer_program": "Mazgāšanas programma",
    "program_placeholder": "Atlasiet Programma",
    "duration": "Ilgums",
    "minutes": "min",
    "time_remaining": "Atlikušais laiks",
    "no_prediction": "Nav prognožu",
    "cycle_in_progress": "Notiek cikls",
    "status": "Statuss",
    "progress": "Progress",
    "select_program": "Izvēlieties programmu, lai skatītu detalizētu informāciju",
    "title": "Nosaukums",
    "status_entity": "Statusa entītija",
    "icon": "Ikona",
    "active_color": "Aktīvās ikonas krāsa",
    "show_state": "Rādīt stāvokli",
    "show_program": "Rādīt programmu",
    "show_details": "Rādīt detaļas",
    "spin_icon": "Griešanās ikona (skrienot)",
    "program_entity": "Programmas entītija",
    "pct_entity": "Progresa entītija (neobligāti)",
    "time_entity": "Laika entītija (neobligāti)",
    "display_mode": "Displeja režīms",
    "show_time_remaining": "Rādīt atlikušo laiku",
    "show_percentage": "Rādīt procentus",
    "entity_not_found": "Entītija nav atrasta"
  },
  "mk": {
    "washer_program": "Програма за перење",
    "program_placeholder": "Изберете Програма",
    "duration": "Времетраење",
    "minutes": "мин",
    "time_remaining": "Преостанато време",
    "no_prediction": "Без предвидување",
    "cycle_in_progress": "Циклус во тек",
    "status": "Статус",
    "progress": "Напредок",
    "select_program": "Изберете програма за да видите детали",
    "title": "Наслов",
    "status_entity": "Статусен ентитет",
    "icon": "Икона",
    "active_color": "Активна боја на иконата",
    "show_state": "Прикажи држава",
    "show_program": "Прикажи програма",
    "show_details": "Прикажи детали",
    "spin_icon": "Икона за вртење (додека работи)",
    "program_entity": "Програмски ентитет",
    "pct_entity": "Ентитет за напредок (изборно)",
    "time_entity": "Временски ентитет (изборно)",
    "display_mode": "Режим на прикажување",
    "show_time_remaining": "Прикажи преостанатото време",
    "show_percentage": "Прикажи процент",
    "entity_not_found": "Субјектот не е пронајден"
  },
  "ml": {
    "washer_program": "വാഷർ പ്രോഗ്രാം",
    "program_placeholder": "പ്രോഗ്രാം തിരഞ്ഞെടുക്കുക",
    "duration": "ദൈർഘ്യം",
    "minutes": "മിനിറ്റ്",
    "time_remaining": "ശേഷിക്കുന്ന സമയം",
    "no_prediction": "പ്രവചനമില്ല",
    "cycle_in_progress": "സൈക്കിൾ പുരോഗമിക്കുന്നു",
    "status": "നില",
    "progress": "പുരോഗതി",
    "select_program": "വിശദാംശങ്ങൾ കാണുന്നതിന് ഒരു പ്രോഗ്രാം തിരഞ്ഞെടുക്കുക",
    "title": "തലക്കെട്ട്",
    "status_entity": "സ്റ്റാറ്റസ് എൻ്റിറ്റി",
    "icon": "ഐക്കൺ",
    "active_color": "സജീവ ഐക്കൺ നിറം",
    "show_state": "സംസ്ഥാനം കാണിക്കുക",
    "show_program": "പ്രോഗ്രാം കാണിക്കുക",
    "show_details": "വിശദാംശങ്ങൾ കാണിക്കുക",
    "spin_icon": "സ്പിന്നിംഗ് ഐക്കൺ (ഓടുമ്പോൾ)",
    "program_entity": "പ്രോഗ്രാം എൻ്റിറ്റി",
    "pct_entity": "പ്രോഗ്രസ് എൻ്റിറ്റി (ഓപ്ഷണൽ)",
    "time_entity": "സമയ എൻ്റിറ്റി (ഓപ്ഷണൽ)",
    "display_mode": "ഡിസ്പ്ലേ മോഡ്",
    "show_time_remaining": "ശേഷിക്കുന്ന സമയം കാണിക്കുക",
    "show_percentage": "ശതമാനം കാണിക്കുക",
    "entity_not_found": "എൻ്റിറ്റി കണ്ടെത്തിയില്ല"
  },
  "nb": {
    "washer_program": "Vaskeprogram",
    "program_placeholder": "Velg Program",
    "duration": "Varighet",
    "minutes": "min",
    "time_remaining": "Gjenstående tid",
    "no_prediction": "Ingen prediksjon",
    "cycle_in_progress": "Syklus pågår",
    "status": "Status",
    "progress": "Framgang",
    "select_program": "Velg et program for å se detaljer",
    "title": "Tittel",
    "status_entity": "Status Entitet",
    "icon": "Ikon",
    "active_color": "Aktiv ikonfarge",
    "show_state": "Vis tilstand",
    "show_program": "Vis program",
    "show_details": "Vis detaljer",
    "spin_icon": "Spinning-ikon (mens du løper)",
    "program_entity": "Program Entitet",
    "pct_entity": "Fremdriftsenhet (valgfritt)",
    "time_entity": "Tidsenhet (valgfritt)",
    "display_mode": "Visningsmodus",
    "show_time_remaining": "Vis gjenværende tid",
    "show_percentage": "Vis prosentandel",
    "entity_not_found": "Enheten ble ikke funnet"
  },
  "nl": {
    "washer_program": "Wasprogramma",
    "program_placeholder": "Selecteer Programma",
    "duration": "Duur",
    "minutes": "min",
    "time_remaining": "Resterende tijd",
    "no_prediction": "Geen voorspelling",
    "cycle_in_progress": "Cyclus in uitvoering",
    "status": "Status",
    "progress": "Voortgang",
    "select_program": "Selecteer een programma om details te bekijken",
    "title": "Titel",
    "status_entity": "Statusentiteit",
    "icon": "Icon",
    "active_color": "Actieve pictogramkleur",
    "show_state": "Toon staat",
    "show_program": "Programma weergeven",
    "show_details": "Details tonen",
    "spin_icon": "Draaiend pictogram (tijdens hardlopen)",
    "program_entity": "Programma-entiteit",
    "pct_entity": "Voortgangsentiteit (optioneel)",
    "time_entity": "Tijdsentiteit (optioneel)",
    "display_mode": "Weergavemodus",
    "show_time_remaining": "Resterende tijd weergeven",
    "show_percentage": "Percentage weergeven",
    "entity_not_found": "Entiteit niet gevonden"
  },
  "nn": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "pl": {
    "washer_program": "Program prania",
    "program_placeholder": "Wybierz Program",
    "duration": "Czas trwania",
    "minutes": "min",
    "time_remaining": "Pozostały czas",
    "no_prediction": "Brak przewidywania",
    "cycle_in_progress": "Cykl w toku",
    "status": "Status",
    "progress": "Postęp",
    "select_program": "Wybierz program, aby zobaczyć szczegóły",
    "title": "Tytuł",
    "status_entity": "Jednostka statusowa",
    "icon": "Ikona",
    "active_color": "Aktywny kolor ikony",
    "show_state": "Pokaż stan",
    "show_program": "Pokaż program",
    "show_details": "Pokaż szczegóły",
    "spin_icon": "Ikona obracania się (podczas biegu)",
    "program_entity": "Jednostka programu",
    "pct_entity": "Jednostka postępu (opcjonalnie)",
    "time_entity": "Jednostka czasu (opcjonalnie)",
    "display_mode": "Tryb wyświetlania",
    "show_time_remaining": "Pokaż pozostały czas",
    "show_percentage": "Pokaż procent",
    "entity_not_found": "Nie znaleziono elementu"
  },
  "pt": {
    "washer_program": "Programa de lavadora",
    "program_placeholder": "Selecione o programa",
    "duration": "Duração",
    "minutes": "min",
    "time_remaining": "Tempo restante",
    "no_prediction": "Sem previsão",
    "cycle_in_progress": "Ciclo em andamento",
    "status": "Status",
    "progress": "Progresso",
    "select_program": "Selecione um programa para ver detalhes",
    "title": "Título",
    "status_entity": "Entidade de status",
    "icon": "Ícone",
    "active_color": "Cor do ícone ativo",
    "show_state": "Mostrar estado",
    "show_program": "Mostrar programa",
    "show_details": "Mostrar detalhes",
    "spin_icon": "Ícone giratório (durante a execução)",
    "program_entity": "Entidade do Programa",
    "pct_entity": "Entidade de progresso (opcional)",
    "time_entity": "Entidade de tempo (opcional)",
    "display_mode": "Modo de exibição",
    "show_time_remaining": "Mostrar tempo restante",
    "show_percentage": "Mostrar porcentagem",
    "entity_not_found": "Entidade não encontrada"
  },
  "pt-BR": {
    "washer_program": "Programa de lavadora",
    "program_placeholder": "Selecione o programa",
    "duration": "Duração",
    "minutes": "min",
    "time_remaining": "Tempo restante",
    "no_prediction": "Sem previsão",
    "cycle_in_progress": "Ciclo em andamento",
    "status": "Status",
    "progress": "Progresso",
    "select_program": "Selecione um programa para ver detalhes",
    "title": "Título",
    "status_entity": "Entidade de status",
    "icon": "Ícone",
    "active_color": "Cor do ícone ativo",
    "show_state": "Mostrar estado",
    "show_program": "Mostrar programa",
    "show_details": "Mostrar detalhes",
    "spin_icon": "Ícone giratório (durante a execução)",
    "program_entity": "Entidade do Programa",
    "pct_entity": "Entidade de progresso (opcional)",
    "time_entity": "Entidade de tempo (opcional)",
    "display_mode": "Modo de exibição",
    "show_time_remaining": "Mostrar tempo restante",
    "show_percentage": "Mostrar porcentagem",
    "entity_not_found": "Entidade não encontrada"
  },
  "ro": {
    "washer_program": "Program de spălat",
    "program_placeholder": "Selectați Program",
    "duration": "Durată",
    "minutes": "min",
    "time_remaining": "Timp rămas",
    "no_prediction": "Fără predicție",
    "cycle_in_progress": "Ciclu în curs",
    "status": "Stare",
    "progress": "Progres",
    "select_program": "Selectați un program pentru a vedea detalii",
    "title": "Titlu",
    "status_entity": "Entitate de stare",
    "icon": "Pictogramă",
    "active_color": "Culoarea pictogramei active",
    "show_state": "Arată stare",
    "show_program": "Arată programul",
    "show_details": "Afișați detalii",
    "spin_icon": "Pictogramă care se învârte (în timpul alergării)",
    "program_entity": "Entitatea de program",
    "pct_entity": "Entitate de progres (opțional)",
    "time_entity": "Entitate oră (Opțional)",
    "display_mode": "Modul de afișare",
    "show_time_remaining": "Arată timpul rămas",
    "show_percentage": "Arată procentul",
    "entity_not_found": "Entitatea nu a fost găsită"
  },
  "ru": {
    "washer_program": "Программа стирки",
    "program_placeholder": "Выберите программу",
    "duration": "Продолжительность",
    "minutes": "мин",
    "time_remaining": "Оставшееся время",
    "no_prediction": "Нет прогноза",
    "cycle_in_progress": "Цикл в процессе",
    "status": "Статус",
    "progress": "Прогресс",
    "select_program": "Выберите программу, чтобы увидеть подробности",
    "title": "Заголовок",
    "status_entity": "Статус объекта",
    "icon": "Икона",
    "active_color": "Цвет активного значка",
    "show_state": "Показать состояние",
    "show_program": "Шоу-программа",
    "show_details": "Показать детали",
    "spin_icon": "Вращающийся значок (во время бега)",
    "program_entity": "Программный объект",
    "pct_entity": "Сущность прогресса (необязательно)",
    "time_entity": "Сущность времени (необязательно)",
    "display_mode": "Режим отображения",
    "show_time_remaining": "Показать оставшееся время",
    "show_percentage": "Показать процент",
    "entity_not_found": "Объект не найден"
  },
  "sk": {
    "washer_program": "Program práčky",
    "program_placeholder": "Vyberte položku Program",
    "duration": "Trvanie",
    "minutes": "min",
    "time_remaining": "Zostávajúci čas",
    "no_prediction": "Žiadna predpoveď",
    "cycle_in_progress": "Prebiehajúci cyklus",
    "status": "Stav",
    "progress": "Pokrok",
    "select_program": "Ak chcete zobraziť podrobnosti, vyberte program",
    "title": "Názov",
    "status_entity": "Stavová entita",
    "icon": "Ikona",
    "active_color": "Farba aktívnej ikony",
    "show_state": "Zobraziť stav",
    "show_program": "Zobraziť program",
    "show_details": "Zobraziť podrobnosti",
    "spin_icon": "Ikona otáčania (pri behu)",
    "program_entity": "Programová entita",
    "pct_entity": "Entita pokroku (voliteľné)",
    "time_entity": "Časová entita (voliteľné)",
    "display_mode": "Režim zobrazenia",
    "show_time_remaining": "Zobraziť zostávajúci čas",
    "show_percentage": "Zobraziť percento",
    "entity_not_found": "Entita sa nenašla"
  },
  "sl": {
    "washer_program": "Program za pranje",
    "program_placeholder": "Izberite Program",
    "duration": "Trajanje",
    "minutes": "min",
    "time_remaining": "Preostali čas",
    "no_prediction": "Brez napovedi",
    "cycle_in_progress": "Cikel v teku",
    "status": "Stanje",
    "progress": "Napredek",
    "select_program": "Za ogled podrobnosti izberite program",
    "title": "Naslov",
    "status_entity": "Statusna entiteta",
    "icon": "Ikona",
    "active_color": "Barva aktivne ikone",
    "show_state": "Prikaži stanje",
    "show_program": "Show Program",
    "show_details": "Pokaži podrobnosti",
    "spin_icon": "Vrteča se ikona (med tekom)",
    "program_entity": "Programska entiteta",
    "pct_entity": "Entiteta napredka (neobvezno)",
    "time_entity": "Časovna entiteta (neobvezno)",
    "display_mode": "Način prikaza",
    "show_time_remaining": "Prikaži preostali čas",
    "show_percentage": "Pokaži odstotek",
    "entity_not_found": "Entiteta ni najdena"
  },
  "sq": {
    "washer_program": "Programi i larës",
    "program_placeholder": "Zgjidhni Programin",
    "duration": "Kohëzgjatja",
    "minutes": "min",
    "time_remaining": "Koha e mbetur",
    "no_prediction": "Asnjë Parashikim",
    "cycle_in_progress": "Cikli në vazhdim",
    "status": "Statusi",
    "progress": "Përparim",
    "select_program": "Zgjidhni një program për të parë detajet",
    "title": "Titulli",
    "status_entity": "Entiteti i statusit",
    "icon": "Ikona",
    "active_color": "Ngjyra e ikonës aktive",
    "show_state": "Trego shtetin",
    "show_program": "Shfaq programin",
    "show_details": "Shfaq Detajet",
    "spin_icon": "Ikona rrotulluese (Gjatë funksionimit)",
    "program_entity": "Subjekti i programit",
    "pct_entity": "Entiteti i progresit (opsionale)",
    "time_entity": "Entiteti i kohës (opsionale)",
    "display_mode": "Modaliteti i shfaqjes",
    "show_time_remaining": "Shfaq kohën e mbetur",
    "show_percentage": "Shfaq përqindjen",
    "entity_not_found": "Subjekti nuk u gjet"
  },
  "sr": {
    "washer_program": "Washer Program",
    "program_placeholder": "Select Program",
    "duration": "Duration",
    "minutes": "min",
    "time_remaining": "Time Remaining",
    "no_prediction": "No Prediction",
    "cycle_in_progress": "Cycle in progress",
    "status": "Status",
    "progress": "Progress",
    "select_program": "Select a program to see details",
    "title": "Title",
    "status_entity": "Status Entity",
    "icon": "Icon",
    "active_color": "Active Icon Color",
    "show_state": "Show State",
    "show_program": "Show Program",
    "show_details": "Show Details",
    "spin_icon": "Spinning Icon (While running)",
    "program_entity": "Program Entity",
    "pct_entity": "Progress Entity (Optional)",
    "time_entity": "Time Entity (Optional)",
    "display_mode": "Display Mode",
    "show_time_remaining": "Show Time Remaining",
    "show_percentage": "Show Percentage",
    "entity_not_found": "Entity not found"
  },
  "sr-Latn": {
    "washer_program": "Програм за прање",
    "program_placeholder": "Изаберите Програм",
    "duration": "Трајање",
    "minutes": "мин",
    "time_remaining": "Преостало време",
    "no_prediction": "Без предвиђања",
    "cycle_in_progress": "Циклус је у току",
    "status": "Статус",
    "progress": "Напредак",
    "select_program": "Изаберите програм да бисте видели детаље",
    "title": "Наслов",
    "status_entity": "Статус Ентитета",
    "icon": "Икона",
    "active_color": "Активна боја иконе",
    "show_state": "Прикажи државу",
    "show_program": "Схов Програм",
    "show_details": "Прикажи детаље",
    "spin_icon": "Икона која се окреће (док трчи)",
    "program_entity": "Програм Ентите",
    "pct_entity": "Ентитет напретка (опционо)",
    "time_entity": "Временски ентитет (опционо)",
    "display_mode": "Режим приказа",
    "show_time_remaining": "Прикажи преостало време",
    "show_percentage": "Прикажи проценат",
    "entity_not_found": "Ентитет није пронађен"
  },
  "sv": {
    "washer_program": "Tvättprogram",
    "program_placeholder": "Välj Program",
    "duration": "Varaktighet",
    "minutes": "min",
    "time_remaining": "Återstående tid",
    "no_prediction": "Ingen förutsägelse",
    "cycle_in_progress": "Cykel pågår",
    "status": "Status",
    "progress": "Framsteg",
    "select_program": "Välj ett program för att se detaljer",
    "title": "Titel",
    "status_entity": "Status Entitet",
    "icon": "Ikon",
    "active_color": "Aktiv ikonfärg",
    "show_state": "Visa tillstånd",
    "show_program": "Visa program",
    "show_details": "Visa detaljer",
    "spin_icon": "Spinning-ikon (medan du springer)",
    "program_entity": "Program Entitet",
    "pct_entity": "Progress Entity (valfritt)",
    "time_entity": "Tidsenhet (valfritt)",
    "display_mode": "Visningsläge",
    "show_time_remaining": "Visa återstående tid",
    "show_percentage": "Visa procent",
    "entity_not_found": "Enheten hittades inte"
  },
  "ta": {
    "washer_program": "வாஷர் திட்டம்",
    "program_placeholder": "நிரலைத் தேர்ந்தெடுக்கவும்",
    "duration": "கால அளவு",
    "minutes": "நிமிடம்",
    "time_remaining": "மீதமுள்ள நேரம்",
    "no_prediction": "கணிப்பு இல்லை",
    "cycle_in_progress": "சுழற்சி நடந்து கொண்டிருக்கிறது",
    "status": "நிலை",
    "progress": "முன்னேற்றம்",
    "select_program": "விவரங்களைப் பார்க்க ஒரு நிரலைத் தேர்ந்தெடுக்கவும்",
    "title": "தலைப்பு",
    "status_entity": "நிலை நிறுவனம்",
    "icon": "ஐகான்",
    "active_color": "செயலில் உள்ள ஐகான் நிறம்",
    "show_state": "மாநிலத்தைக் காட்டு",
    "show_program": "நிகழ்ச்சி நிரல்",
    "show_details": "விவரங்களைக் காட்டு",
    "spin_icon": "ஸ்பின்னிங் ஐகான் (இயங்கும் போது)",
    "program_entity": "நிரல் நிறுவனம்",
    "pct_entity": "முன்னேற்ற நிறுவனம் (விரும்பினால்)",
    "time_entity": "நேர பொருள் (விரும்பினால்)",
    "display_mode": "காட்சி முறை",
    "show_time_remaining": "மீதமுள்ள நேரத்தைக் காட்டு",
    "show_percentage": "சதவீதத்தைக் காட்டு",
    "entity_not_found": "பொருள் கிடைக்கவில்லை"
  },
  "te": {
    "washer_program": "వాషర్ ప్రోగ్రామ్",
    "program_placeholder": "ప్రోగ్రామ్‌ని ఎంచుకోండి",
    "duration": "వ్యవధి",
    "minutes": "నిమి",
    "time_remaining": "సమయం మిగిలి ఉంది",
    "no_prediction": "అంచనా లేదు",
    "cycle_in_progress": "చక్రం పురోగతిలో ఉంది",
    "status": "స్థితి",
    "progress": "పురోగతి",
    "select_program": "వివరాలను చూడటానికి ప్రోగ్రామ్‌ను ఎంచుకోండి",
    "title": "శీర్షిక",
    "status_entity": "స్టేటస్ ఎంటిటీ",
    "icon": "చిహ్నం",
    "active_color": "సక్రియ చిహ్నం రంగు",
    "show_state": "రాష్ట్రాన్ని చూపించు",
    "show_program": "కార్యక్రమం చూపించు",
    "show_details": "వివరాలను చూపించు",
    "spin_icon": "స్పిన్నింగ్ ఐకాన్ (నడుస్తున్నప్పుడు)",
    "program_entity": "ప్రోగ్రామ్ ఎంటిటీ",
    "pct_entity": "ప్రోగ్రెస్ ఎంటిటీ (ఐచ్ఛికం)",
    "time_entity": "టైమ్ ఎంటిటీ (ఐచ్ఛికం)",
    "display_mode": "ప్రదర్శన మోడ్",
    "show_time_remaining": "మిగిలిన సమయాన్ని చూపించు",
    "show_percentage": "శాతాన్ని చూపించు",
    "entity_not_found": "ఎంటిటీ కనుగొనబడలేదు"
  },
  "th": {
    "washer_program": "โปรแกรมเครื่องซักผ้า",
    "program_placeholder": "เลือกโปรแกรม",
    "duration": "ระยะเวลา",
    "minutes": "นาที",
    "time_remaining": "เวลาที่เหลืออยู่",
    "no_prediction": "ไม่มีการคาดการณ์",
    "cycle_in_progress": "อยู่ระหว่างดำเนินการ",
    "status": "สถานะ",
    "progress": "ความคืบหน้า",
    "select_program": "เลือกโปรแกรมเพื่อดูรายละเอียด",
    "title": "ชื่อ",
    "status_entity": "เอนทิตีสถานะ",
    "icon": "ไอคอน",
    "active_color": "สีไอคอนที่ใช้งานอยู่",
    "show_state": "แสดงสถานะ",
    "show_program": "โปรแกรมโชว์",
    "show_details": "แสดงรายละเอียด",
    "spin_icon": "ไอคอนหมุน (ขณะวิ่ง)",
    "program_entity": "เอนทิตีของโปรแกรม",
    "pct_entity": "เอนทิตีความคืบหน้า (ไม่บังคับ)",
    "time_entity": "เอนทิตีเวลา (ไม่บังคับ)",
    "display_mode": "โหมดการแสดงผล",
    "show_time_remaining": "แสดงเวลาที่เหลืออยู่",
    "show_percentage": "แสดงเปอร์เซ็นต์",
    "entity_not_found": "ไม่พบเอนทิตี"
  },
  "tr": {
    "washer_program": "Yıkama Programı",
    "program_placeholder": "Program Seç",
    "duration": "Süre",
    "minutes": "dk.",
    "time_remaining": "Kalan Süre",
    "no_prediction": "Tahmin Yok",
    "cycle_in_progress": "Döngü devam ediyor",
    "status": "Durum",
    "progress": "İlerlemek",
    "select_program": "Ayrıntıları görmek için bir program seçin",
    "title": "Başlık",
    "status_entity": "Durum Varlığı",
    "icon": "Simge",
    "active_color": "Etkin Simge Rengi",
    "show_state": "Durumu Göster",
    "show_program": "Programı Göster",
    "show_details": "Ayrıntıları Göster",
    "spin_icon": "Dönen Simge (Koşarken)",
    "program_entity": "Program Varlığı",
    "pct_entity": "İlerleme Varlığı (İsteğe bağlı)",
    "time_entity": "Zaman Varlığı (İsteğe Bağlı)",
    "display_mode": "Ekran Modu",
    "show_time_remaining": "Kalan Süreyi Göster",
    "show_percentage": "Yüzdeyi Göster",
    "entity_not_found": "Varlık bulunamadı"
  },
  "uk": {
    "washer_program": "Програма прання",
    "program_placeholder": "Виберіть програму",
    "duration": "Тривалість",
    "minutes": "хв",
    "time_remaining": "Час, що залишився",
    "no_prediction": "Без передбачення",
    "cycle_in_progress": "Цикл триває",
    "status": "Статус",
    "progress": "Прогрес",
    "select_program": "Виберіть програму, щоб переглянути деталі",
    "title": "Назва",
    "status_entity": "Status Entity",
    "icon": "значок",
    "active_color": "Активний колір значка",
    "show_state": "Показати стан",
    "show_program": "Шоу програма",
    "show_details": "Показати деталі",
    "spin_icon": "Піктограма обертання (під час бігу)",
    "program_entity": "Програмна сутність",
    "pct_entity": "Сутність прогресу (необов'язково)",
    "time_entity": "Сутність часу (необов’язково)",
    "display_mode": "Режим відображення",
    "show_time_remaining": "Показати час, що залишився",
    "show_percentage": "Показати відсоток",
    "entity_not_found": "Об'єкт не знайдено"
  },
  "ur": {
    "washer_program": "واشر پروگرام",
    "program_placeholder": "پروگرام منتخب کریں۔",
    "duration": "دورانیہ",
    "minutes": "منٹ",
    "time_remaining": "باقی وقت",
    "no_prediction": "کوئی پیشین گوئی نہیں۔",
    "cycle_in_progress": "سائیکل جاری ہے۔",
    "status": "حیثیت",
    "progress": "پیش رفت",
    "select_program": "تفصیلات دیکھنے کے لیے ایک پروگرام منتخب کریں۔",
    "title": "عنوان",
    "status_entity": "اسٹیٹس ہستی",
    "icon": "آئیکن",
    "active_color": "ایکٹو آئیکن کا رنگ",
    "show_state": "ریاست دکھائیں۔",
    "show_program": "پروگرام دکھائیں۔",
    "show_details": "تفصیلات دکھائیں۔",
    "spin_icon": "گھومنے کا آئیکن (چلتے وقت)",
    "program_entity": "پروگرام ہستی",
    "pct_entity": "پیش رفت ہستی (اختیاری)",
    "time_entity": "وقت کی ہستی (اختیاری)",
    "display_mode": "ڈسپلے موڈ",
    "show_time_remaining": "باقی وقت دکھائیں۔",
    "show_percentage": "فیصد دکھائیں۔",
    "entity_not_found": "ہستی نہیں ملی"
  },
  "vi": {
    "washer_program": "Chương trình máy giặt",
    "program_placeholder": "Chọn chương trình",
    "duration": "Khoảng thời gian",
    "minutes": "phút",
    "time_remaining": "Thời gian còn lại",
    "no_prediction": "Không có dự đoán",
    "cycle_in_progress": "Đang tiến hành chu kỳ",
    "status": "Trạng thái",
    "progress": "Tiến triển",
    "select_program": "Chọn chương trình để xem chi tiết",
    "title": "Tiêu đề",
    "status_entity": "Thực thể trạng thái",
    "icon": "Biểu tượng",
    "active_color": "Màu biểu tượng hoạt động",
    "show_state": "Hiển thị trạng thái",
    "show_program": "Hiển thị chương trình",
    "show_details": "Hiển thị chi tiết",
    "spin_icon": "Biểu tượng quay (Trong khi chạy)",
    "program_entity": "Thực thể chương trình",
    "pct_entity": "Thực thể tiến độ (Tùy chọn)",
    "time_entity": "Thực thể thời gian (Tùy chọn)",
    "display_mode": "Chế độ hiển thị",
    "show_time_remaining": "Hiển thị thời gian còn lại",
    "show_percentage": "Hiển thị phần trăm",
    "entity_not_found": "Không tìm thấy thực thể"
  },
  "zh-Hans": {
    "washer_program": "清洗程序",
    "program_placeholder": "选择节目",
    "duration": "期间",
    "minutes": "分钟",
    "time_remaining": "剩余时间",
    "no_prediction": "没有预测",
    "cycle_in_progress": "循环正在进行中",
    "status": "地位",
    "progress": "进步",
    "select_program": "选择一个程序以查看详细信息",
    "title": "标题",
    "status_entity": "状态实体",
    "icon": "图标",
    "active_color": "活动图标颜色",
    "show_state": "显示状态",
    "show_program": "演出节目",
    "show_details": "显示详情",
    "spin_icon": "旋转图标（运行时）",
    "program_entity": "程序实体",
    "pct_entity": "进度实体（可选）",
    "time_entity": "时间实体（可选）",
    "display_mode": "显示模式",
    "show_time_remaining": "显示剩余时间",
    "show_percentage": "显示百分比",
    "entity_not_found": "未找到实体"
  },
  "zh-Hant": {
    "washer_program": "清洗程序",
    "program_placeholder": "選擇節目",
    "duration": "期間",
    "minutes": "分分鐘",
    "time_remaining": "剩餘時間",
    "no_prediction": "沒有預測",
    "cycle_in_progress": "循環正在進行中",
    "status": "地位",
    "progress": "進步",
    "select_program": "選擇一個程序以查看詳細信息",
    "title": "標題",
    "status_entity": "狀態實體",
    "icon": "圖示",
    "active_color": "活動圖示顏色",
    "show_state": "顯示狀態",
    "show_program": "演出節目",
    "show_details": "顯示詳情",
    "spin_icon": "旋轉圖示（運行時）",
    "program_entity": "程式實體",
    "pct_entity": "進度實體（可選）",
    "time_entity": "時間實體（可選）",
    "display_mode": "顯示模式",
    "show_time_remaining": "顯示剩餘時間",
    "show_percentage": "顯示百分比",
    "entity_not_found": "未找到實體"
  }
};

class WashDataCard extends HTMLElement {
  _resolveLanguage() {
    const raw =
      (this._hass && this._hass.locale && this._hass.locale.language) ||
      (this._hass && this._hass.language) ||
      "en";
    if (!raw || typeof raw !== "string") return "en";
    return raw;
  }

  static getStubConfig() {
    return {
      entity: "sensor.washing_machine_state",
      title: "Washing Machine",
      icon: "mdi:washing-machine",
      display_mode: "time",
      active_color: [33, 150, 243],
      show_state: true,
      show_program: true,
      show_details: true,
      spin_icon: true
    };
  }

  static getConfigElement() {
    return document.createElement(EDITOR_TAG);
  }

  _getTranslation(key) {
    const lang = this._resolveLanguage();
    const baseLang = lang.split("-")[0];
    const translations = TRANSLATIONS[lang] || TRANSLATIONS[baseLang] || TRANSLATIONS["en"];
    return translations[key] || TRANSLATIONS["en"][key] || key;
  }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._rendered = false;
    this._handleClick = this._handleClick.bind(this);
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error("Please define an entity");
    }
    this._cfg = { ...WashDataCard.getStubConfig(), ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._update();
  }

  getCardSize() {
    return 1;
  }

  _handleClick() {
    const entityId = this._cfg.entity;
    const event = new CustomEvent("hass-more-info", {
      detail: { entityId },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }

  _render() {
    if (!this.shadowRoot) return;

    // Only create the DOM once to avoid memory leaks from duplicate event listeners
    if (!this._rendered) {
      this.shadowRoot.innerHTML = `
        <style>
          :host {
            display: block;
            height: 100%;
          }
          ha-card {
            padding: 0;
            background: var(--ha-card-background, var(--card-background-color, white));
            border-radius: var(--ha-card-border-radius, 12px);
            box-shadow: var(--ha-card-box-shadow, none);
            overflow: hidden;
            cursor: pointer;
            height: 100%;
            display: flex;
            align-items: center;
            box-sizing: border-box;
            border: var(--ha-card-border-width, 1px) solid var(--ha-card-border-color, var(--divider-color));
          }
          .tile {
            display: flex;
            flex-direction: row;
            align-items: center;
            padding: 0 12px;
            gap: 12px;
            width: 100%;
            height: 100%;
            min-height: 56px; /* standard tile height */
            max-height: 56px;
            box-sizing: border-box;
          }
          .icon-container {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--tile-icon-bg, rgba(128, 128, 128, 0.1));
            color: var(--tile-icon-color, var(--primary-text-color));
            flex-shrink: 0;
            transition: background-color 0.3s, color 0.3s;
          }
          ha-icon {
            --mdc-icon-size: 24px;
          }
          .info {
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: hidden;
            flex: 1;
          }
          .primary {
            font-weight: 500;
            font-size: 14px;
            color: var(--primary-text-color);
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            line-height: 1.2;
          }
          .secondary {
            font-size: 12px;
            color: var(--secondary-text-color);
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            line-height: 1.2;
            margin-top: 2px;
          }
          
          /* Animation */
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .spinning {
            animation: spin 2s linear infinite;
          }
        </style>
        <ha-card id="card">
          <div class="tile">
            <div class="icon-container" id="icon-container">
              <ha-icon id="icon"></ha-icon>
            </div>
            <div class="info">
              <div class="primary" id="title"></div>
              <div class="secondary" id="state"></div>
            </div>
          </div>
        </ha-card>
      `;

      this.shadowRoot.getElementById("card").addEventListener("click", this._handleClick);
      this._rendered = true;
    }

    this._update();
  }

  _update() {
    if (!this.shadowRoot || !this._hass || !this._cfg) return;

    const entityId = this._cfg.entity;
    const stateObj = this._hass.states[entityId];

    const titleEl = this.shadowRoot.getElementById("title");
    const stateEl = this.shadowRoot.getElementById("state");
    const iconEl = this.shadowRoot.getElementById("icon");
    const iconContainer = this.shadowRoot.getElementById("icon-container");

    if (!stateObj) {
      if (titleEl) titleEl.textContent = this._getTranslation("entity_not_found");
      if (stateEl) stateEl.textContent = entityId;
      return;
    }

    const title = this._cfg.title || this._getTranslation("washing_machine");
    const icon = this._cfg.icon || stateObj.attributes.icon || "mdi:washing-machine";
    const activeColor = this._cfg.active_color;

    const state = stateObj.state;
    // Treat as inactive if off, unknown, unavailable, idle
    const isInactive = ['off', 'unknown', 'unavailable', 'idle'].includes(state.toLowerCase());

    if (isInactive) {
      iconContainer.style.background = `rgba(128, 128, 128, 0.1)`;
      iconContainer.style.color = `var(--disabled-text-color, grey)`;
    } else {
      let colorCss = "var(--primary-color)";
      let bgCss = "rgba(var(--rgb-primary-color, 33, 150, 243), 0.2)";

      if (Array.isArray(activeColor)) {
        const [r, g, b] = activeColor;
        colorCss = `rgb(${r}, ${g}, ${b})`;
        bgCss = `rgba(${r}, ${g}, ${b}, 0.2)`;
      } else if (activeColor) {
        colorCss = activeColor;
        bgCss = `rgba(128, 128, 128, 0.15)`;
      }

      iconContainer.style.color = colorCss;
      iconContainer.style.background = bgCss;
    }

    iconEl.setAttribute("icon", icon);
    if (state.toLowerCase() === 'running' && this._cfg.spin_icon !== false) {
      iconEl.classList.add("spinning");
    } else {
      iconEl.classList.remove("spinning");
    }
    titleEl.textContent = title;

    const attr = stateObj.attributes;
    const parts = [];

    // 1. State / Sub-State
    // Default show_state to true if undefined
    if (this._cfg.show_state !== false) {
      if (state.toLowerCase() === 'running') {
        const subState = attr.sub_state;
        if (subState) {
          // If sub_state is "Running (Rinsing)", extract "Rinsing"
          const match = subState.match(/Running \((.*)\)/);
          if (match && match[1]) {
            parts.push(match[1]);
          } else {
            parts.push(subState);
          }
        }
        // If no sub_state (or just "Running"), we show NOTHING (redundant)
      } else {
        // Not running (e.g. Off, Completed, etc) - show standard state
        parts.push(state.charAt(0).toUpperCase() + state.slice(1));
      }
    }

    // 2. Program
    if (this._cfg.show_program !== false) {
      let program = "";
      if (this._cfg.program_entity) {
        const progState = this._hass.states[this._cfg.program_entity];
        if (progState) program = progState.state;
      } else if (attr.program) {
        program = attr.program;
      }
      if (program && !["unknown", "none", "off", "unavailable"].includes(program.toLowerCase())) {
        parts.push(program);
      }
    }

    // 3. Details (Time / Pct)
    if (this._cfg.show_details !== false && !isInactive) {
      let remaining = "";
      if (this._cfg.time_entity) {
        remaining = this._hass.states[this._cfg.time_entity]?.state;
      } else if (attr.time_remaining) {
        remaining = attr.time_remaining;
      }

      let pct = "";
      if (this._cfg.pct_entity) {
        pct = this._hass.states[this._cfg.pct_entity]?.state;
      } else if (attr.cycle_progress) {
        pct = attr.cycle_progress;
      }

      if (this._cfg.display_mode === 'percentage' && pct) {
        parts.push(`${Math.round(pct)}%`);
      } else if (remaining) {
        // Append 'min' if it is a number (WashData attribute is raw minutes)
        if (!isNaN(remaining)) {
          parts.push(`${remaining} ${this._getTranslation("minutes")}`);
        } else {
          parts.push(remaining);
        }
      }
    }

    stateEl.textContent = parts.length > 0 ? parts.join(" • ") : "";
  }
}

class WashDataCardEditor extends HTMLElement {
  _resolveLanguage() {
    const raw =
      (this._hass && this._hass.locale && this._hass.locale.language) ||
      (this._hass && this._hass.language) ||
      "en";
    if (!raw || typeof raw !== "string") return "en";
    return raw;
  }

  _getTranslation(key) {
    const lang = this._resolveLanguage();
    const baseLang = lang.split("-")[0];
    const translations = TRANSLATIONS[lang] || TRANSLATIONS[baseLang] || TRANSLATIONS["en"];
    return translations[key] || TRANSLATIONS["en"][key] || key;
  }

  setConfig(config) {
    this._cfg = { ...WashDataCard.getStubConfig(), ...config };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    if (this._form) {
      this._form.hass = hass;
    }
  }

  _render() {
    if (!this.shadowRoot) {
      this.attachShadow({ mode: "open" });
    }

    if (!this._form) {
      this.shadowRoot.innerHTML = `
        <style>
          .editor-container {
            padding: 16px;
            max-width: 400px; /* Constrain editor width */
          }
          ha-form {
            display: block;
          }
        </style>
        <div class="editor-container" id="editor-container"></div>
      `;
      this._form = document.createElement("ha-form");
      this.shadowRoot.getElementById("editor-container").appendChild(this._form);

      this._form.addEventListener("value-changed", (ev) => this._valueChanged(ev));

      this._form.schema = [
        { name: "title", selector: { text: {} } },
        { name: "entity", selector: { entity: { domain: "sensor" } } },
        { name: "icon", selector: { icon: {} } },
        { name: "active_color", selector: { color_rgb: {} } },
        { name: "show_state", selector: { boolean: {} } },
        { name: "show_program", selector: { boolean: {} } },
        { name: "show_details", selector: { boolean: {} } },
        { name: "spin_icon", selector: { boolean: {} } },
        {
          name: "display_mode",
          selector: {
            select: {
              options: [
                { value: "time", label: this._getTranslation("show_time_remaining") },
                { value: "percentage", label: this._getTranslation("show_percentage") }
              ],
              mode: "dropdown"
            }
          }
        },
        { name: "program_entity", selector: { entity: { domain: ["sensor", "select", "input_select", "input_text"] } } },
        { name: "pct_entity", selector: { entity: { domain: "sensor" } } },
        { name: "time_entity", selector: { entity: { domain: "sensor" } } },
      ];

      this._form.computeLabel = (schema) => {
        const labels = {
          title: this._getTranslation("title"),
          entity: this._getTranslation("status_entity"),
          icon: this._getTranslation("icon"),
          active_color: this._getTranslation("active_color"),
          show_state: this._getTranslation("show_state"),
          show_program: this._getTranslation("show_program"),
          show_details: this._getTranslation("show_details"),
          spin_icon: this._getTranslation("spin_icon"),
          program_entity: this._getTranslation("program_entity"),
          pct_entity: this._getTranslation("pct_entity"),
          time_entity: this._getTranslation("time_entity"),
          display_mode: this._getTranslation("display_mode")
        };
        return labels[schema.name] || schema.name;
      };
    }

    this._form.data = this._cfg;
    if (this._hass) {
      this._form.hass = this._hass;
    }
  }

  _valueChanged(ev) {
    if (!this._cfg || !this._hass) return;
    const val = ev.detail.value;
    this._cfg = { ...this._cfg, ...val };

    const event = new CustomEvent("config-changed", {
      detail: { config: this._cfg },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }
}

customElements.define(CARD_TAG, WashDataCard);
customElements.define(EDITOR_TAG, WashDataCardEditor);

window.customCards = window.customCards || [];
window.customCards.push({
  type: CARD_TAG,
  name: "WashData Tile Card",
  preview: true,
  description: "A compact tile-style card for washing machines.",
});
