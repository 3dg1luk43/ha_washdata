// WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
// Copyright (C) 2026 Lukas Bandura
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
const CARD_TAG = "ha-washdata-card";
const EDITOR_TAG = "ha-washdata-card-editor";

// Gesture timing (ms) and movement tolerance (px) for tap / hold / double-tap,
// chosen to match Home Assistant's own action handler conventions.
const HOLD_MS = 500;
const DOUBLE_TAP_MS = 250;
const TAP_MOVE_TOLERANCE = 10;

const TRANSLATIONS = {"en":{"washer_program":"Washer Program","program_placeholder":"Select Program","duration":"Duration","minutes":"min","time_remaining":"Time Remaining","no_prediction":"No Prediction","cycle_in_progress":"Cycle in progress","status":"Status","progress":"Progress","select_program":"Select a program to see details","title":"Title","status_entity":"Status Entity","icon":"Icon","active_color":"Active Icon Color","show_state":"Show State","show_program":"Show Program","show_details":"Show Details","spin_icon":"Spinning Icon (While running)","program_entity":"Program Entity","pct_entity":"Progress Entity (Optional)","time_entity":"Time Entity (Optional)","display_mode":"Display Mode","show_time_remaining":"Show Time Remaining","show_percentage":"Show Percentage","entity_not_found":"Entity not found","tap_action":"Tap Action","hold_action":"Hold Action","double_tap_action":"Double Tap Action"},"bg":{"washer_program":"Програма за пране","program_placeholder":"Изберете Програма","duration":"Продължителност","minutes":"мин","time_remaining":"Оставащо време","no_prediction":"Няма прогноза","cycle_in_progress":"Цикълът е в ход","status":"Статус","progress":"Напредък","select_program":"Изберете програма, за да видите подробности","title":"Заглавие","status_entity":"Състояние на обекта","icon":"Икона","active_color":"Цвят на активната икона","show_state":"Показване на състояние","show_program":"Шоу програма","show_details":"Показване на подробности","spin_icon":"Въртяща се икона (докато работи)","program_entity":"Програмен субект","pct_entity":"Обект на напредъка (по избор)","time_entity":"Времеви обект (по избор)","display_mode":"Режим на показване","show_time_remaining":"Показване на оставащото време","show_percentage":"Показване на процента","entity_not_found":"Обектът не е намерен","tap_action":"Докосване на действие","hold_action":"Задръж действие","double_tap_action":"Двойно докосване"},"bs":{"washer_program":"Program za pranje","program_placeholder":"Odaberite Program","duration":"Trajanje","minutes":"min","time_remaining":"Preostalo vrijeme","no_prediction":"Nema predviđanja","cycle_in_progress":"Ciklus je u toku","status":"Status","progress":"Napredak","select_program":"Odaberite program da vidite detalje","title":"Naslov","status_entity":"Status Entiteta","icon":"Ikona","active_color":"Aktivna boja ikone","show_state":"Prikaži državu","show_program":"Show Program","show_details":"Prikaži detalje","spin_icon":"Ikona za okretanje (dok trčanje)","program_entity":"Programski entitet","pct_entity":"Entitet napretka (opciono)","time_entity":"Vremenski entitet (opcionalno)","display_mode":"Način prikaza","show_time_remaining":"Prikaži preostalo vrijeme","show_percentage":"Prikaži procenat","entity_not_found":"Entitet nije pronađen","tap_action":"Dodirnite Akcija","hold_action":"Držite akciju","double_tap_action":"Dvostruki dodir Akcija"},"cs":{"washer_program":"Program pračky","program_placeholder":"Vyberte Program","duration":"Trvání","minutes":"min","time_remaining":"Zbývající čas","no_prediction":"Žádná předpověď","cycle_in_progress":"Cyklus probíhá","status":"Postavení","progress":"Pokrok","select_program":"Chcete-li zobrazit podrobnosti, vyberte program","title":"Titul","status_entity":"Stavová entita","icon":"Ikona","active_color":"Barva aktivní ikony","show_state":"Zobrazit stav","show_program":"Zobrazit program","show_details":"Zobrazit podrobnosti","spin_icon":"Ikona rotace (při běhu)","program_entity":"Entita programu","pct_entity":"Entita průběhu (volitelné)","time_entity":"Časová entita (volitelné)","display_mode":"Režim zobrazení","show_time_remaining":"Zobrazit zbývající čas","show_percentage":"Zobrazit procento","entity_not_found":"Entita nenalezena","tap_action":"Klepněte na možnost Akce","hold_action":"Držet akci","double_tap_action":"Akce dvojitého klepnutí"},"da":{"washer_program":"Vaskeprogram","program_placeholder":"Vælg Program","duration":"Varighed","minutes":"min","time_remaining":"Tid tilbage","no_prediction":"Ingen forudsigelse","cycle_in_progress":"Cyklus i gang","status":"Status","progress":"Fremskridt","select_program":"Vælg et program for at se detaljer","title":"Titel","status_entity":"Statusenhed","icon":"Ikon","active_color":"Aktiv ikon farve","show_state":"Vis tilstand","show_program":"Vis program","show_details":"Vis detaljer","spin_icon":"Spinning-ikon (mens du løber)","program_entity":"Programenhed","pct_entity":"Fremskridtsenhed (valgfrit)","time_entity":"Tidsenhed (valgfrit)","display_mode":"Visningstilstand","show_time_remaining":"Vis resterende tid","show_percentage":"Vis procent","entity_not_found":"Enheden blev ikke fundet","tap_action":"Tap på handling","hold_action":"Hold handling","double_tap_action":"Dobbelt tastehandling"},"de":{"washer_program":"Waschprogramm","program_placeholder":"Wählen Sie Programm","duration":"Dauer","minutes":"min","time_remaining":"Verbleibende Zeit","no_prediction":"Keine Vorhersage","cycle_in_progress":"Zyklus läuft","status":"Status","progress":"Fortschritt","select_program":"Wählen Sie ein Programm aus, um Details anzuzeigen","title":"Titel","status_entity":"Status-Entität","icon":"Symbol","active_color":"Aktive Symbolfarbe","show_state":"Status anzeigen","show_program":"Programm anzeigen","show_details":"Details anzeigen","spin_icon":"Spinning-Symbol (während des Laufens)","program_entity":"Programmeinheit","pct_entity":"Fortschrittsentität (optional)","time_entity":"Zeiteinheit (optional)","display_mode":"Anzeigemodus","show_time_remaining":"Verbleibende Zeit anzeigen","show_percentage":"Prozentsatz anzeigen","entity_not_found":"Entität nicht gefunden","tap_action":"Tippen Sie auf","hold_action":"Action spielen","double_tap_action":"Doppeltipp-Aktion"},"el":{"washer_program":"Πρόγραμμα πλύσης","program_placeholder":"Επιλέξτε Πρόγραμμα","duration":"Διάρκεια","minutes":"ελάχ","time_remaining":"Χρόνος που απομένει","no_prediction":"Καμία Πρόβλεψη","cycle_in_progress":"Κύκλος σε εξέλιξη","status":"Κατάσταση","progress":"Πρόοδος","select_program":"Επιλέξτε ένα πρόγραμμα για να δείτε λεπτομέρειες","title":"Τίτλος","status_entity":"Οντότητα κατάστασης","icon":"Εικόνισμα","active_color":"Χρώμα ενεργού εικονιδίου","show_state":"Εμφάνιση κατάστασης","show_program":"Εμφάνιση προγράμματος","show_details":"Εμφάνιση λεπτομερειών","spin_icon":"Περιστρεφόμενο εικονίδιο (Κατά την εκτέλεση)","program_entity":"Οντότητα προγράμματος","pct_entity":"Οντότητα προόδου (Προαιρετικό)","time_entity":"Οντότητα ώρας (Προαιρετικό)","display_mode":"Λειτουργία εμφάνισης","show_time_remaining":"Εμφάνιση χρόνου που απομένει","show_percentage":"Εμφάνιση ποσοστού","entity_not_found":"Η οντότητα δεν βρέθηκε","tap_action":"Πατήστε ενέργεια","hold_action":"Διατήρηση ενέργειας","double_tap_action":"Διπλή ενέργεια πατήματος"},"es":{"washer_program":"Programa de lavadora","program_placeholder":"Seleccionar programa","duration":"Duración","minutes":"mín.","time_remaining":"Tiempo restante","no_prediction":"Sin predicción","cycle_in_progress":"Ciclo en progreso","status":"Estado","progress":"Progreso","select_program":"Selecciona un programa para ver detalles","title":"Título","status_entity":"Entidad de estado","icon":"Icono","active_color":"Color del icono activo","show_state":"Mostrar estado","show_program":"Mostrar programa","show_details":"Mostrar detalles","spin_icon":"Icono de giro (mientras se ejecuta)","program_entity":"Entidad del programa","pct_entity":"Entidad de progreso (opcional)","time_entity":"Entidad de tiempo (opcional)","display_mode":"Modo de visualización","show_time_remaining":"Mostrar tiempo restante","show_percentage":"Mostrar porcentaje","entity_not_found":"Entidad no encontrada","tap_action":"Toque Acción","hold_action":"Mantener acción","double_tap_action":"Doble toque de acción"},"et":{"washer_program":"Pesumasina programm","program_placeholder":"Valige Programm","duration":"Kestus","minutes":"min","time_remaining":"Järelejäänud aeg","no_prediction":"Ei mingit ennustust","cycle_in_progress":"Tsükkel on pooleli","status":"Olek","progress":"Edusammud","select_program":"Üksikasjade vaatamiseks valige programm","title":"Pealkiri","status_entity":"Olekuüksus","icon":"Ikoon","active_color":"Aktiivne ikooni värv","show_state":"Näita olekut","show_program":"Näita programmi","show_details":"Näita üksikasju","spin_icon":"Pöörlev ikoon (jooksmise ajal)","program_entity":"Programmi üksus","pct_entity":"Edenemisüksus (valikuline)","time_entity":"Ajaüksus (valikuline)","display_mode":"Kuvamisrežiim","show_time_remaining":"Näita järelejäänud aega","show_percentage":"Näita protsenti","entity_not_found":"Üksust ei leitud","tap_action":"Puudutustoiming","hold_action":"Hoidke tegevust","double_tap_action":"Topeltpuutetoiming"},"fi":{"washer_program":"Pesuohjelma","program_placeholder":"Valitse Ohjelma","duration":"Kesto","minutes":"min","time_remaining":"Aikaa jäljellä","no_prediction":"Ei ennustetta","cycle_in_progress":"Kierto käynnissä","status":"Tila","progress":"Edistyminen","select_program":"Valitse ohjelma nähdäksesi tiedot","title":"Otsikko","status_entity":"Tilayksikkö","icon":"Kuvake","active_color":"Aktiivinen kuvakkeen väri","show_state":"Näytä tila","show_program":"Näytä ohjelma","show_details":"Näytä tiedot","spin_icon":"Pyörivä kuvake (juoksessa)","program_entity":"Ohjelmakokonaisuus","pct_entity":"Etenemiskokonaisuus (valinnainen)","time_entity":"Aikakokonaisuus (valinnainen)","display_mode":"Näyttötila","show_time_remaining":"Näytä jäljellä oleva aika","show_percentage":"Näytä prosenttiosuus","entity_not_found":"Kokonaisuutta ei löydy","tap_action":"Napauta toimintoa","hold_action":"Pidä toimintoa","double_tap_action":"Kaksoisnapaustoiminto"},"fr":{"washer_program":"Programme de laveuse","program_placeholder":"Sélectionnez le programme","duration":"Durée","minutes":"min","time_remaining":"Temps restant","no_prediction":"Aucune prédiction","cycle_in_progress":"Cycle en cours","status":"Statut","progress":"Progrès","select_program":"Sélectionnez un programme pour voir les détails","title":"Titre","status_entity":"Entité de statut","icon":"Icône","active_color":"Couleur de l'icône active","show_state":"Afficher l'état","show_program":"Programme du spectacle","show_details":"Afficher les détails","spin_icon":"Icône de rotation (pendant l'exécution)","program_entity":"Entité du programme","pct_entity":"Entité de progression (facultatif)","time_entity":"Entité temporelle (facultatif)","display_mode":"Mode d'affichage","show_time_remaining":"Afficher le temps restant","show_percentage":"Afficher le pourcentage","entity_not_found":"Entité introuvable","tap_action":"Appuyez sur Action","hold_action":"Maintenez l'action","double_tap_action":"Double action de la touche"},"hr":{"washer_program":"Program za pranje","program_placeholder":"Odaberite Program","duration":"Trajanje","minutes":"min","time_remaining":"Preostalo vrijeme","no_prediction":"Nema predviđanja","cycle_in_progress":"Ciklus u tijeku","status":"Status","progress":"Napredak","select_program":"Odaberite program da biste vidjeli pojedinosti","title":"Titula","status_entity":"Statusni entitet","icon":"Ikona","active_color":"Boja aktivne ikone","show_state":"Prikaži stanje","show_program":"Show Program","show_details":"Prikaži pojedinosti","spin_icon":"Ikona koja se vrti (dok radi)","program_entity":"Programski entitet","pct_entity":"Entitet napretka (neobavezno)","time_entity":"Entitet vremena (neobavezno)","display_mode":"Način prikaza","show_time_remaining":"Prikaži preostalo vrijeme","show_percentage":"Prikaži postotak","entity_not_found":"Entitet nije pronađen","tap_action":"Dodirnite Akcija","hold_action":"Zadrži akciju","double_tap_action":"Akcija dvostrukog dodira"},"hu":{"washer_program":"Mosó program","program_placeholder":"Válassza a Program lehetőséget","duration":"Időtartam","minutes":"min","time_remaining":"Hátralévő idő","no_prediction":"Nincs előrejelzés","cycle_in_progress":"Ciklus folyamatban","status":"Állapot","progress":"Előrehalad","select_program":"Válasszon ki egy programot a részletek megtekintéséhez","title":"Cím","status_entity":"Állapot entitás","icon":"Ikon","active_color":"Aktív Ikon színe","show_state":"Állapot megjelenítése","show_program":"Program megjelenítése","show_details":"Részletek megjelenítése","spin_icon":"Pörgő ikon (futás közben)","program_entity":"Program entitás","pct_entity":"Haladási entitás (opcionális)","time_entity":"Idő entitás (opcionális)","display_mode":"Kijelző mód","show_time_remaining":"Mutasd a hátralévő időt","show_percentage":"Százalék megjelenítése","entity_not_found":"Az entitás nem található","tap_action":"Koppintson a Művelet elemre","hold_action":"Tartsa akciót","double_tap_action":"Dupla koppintás művelet"},"is":{"washer_program":"Þvottavélaforrit","program_placeholder":"Veldu Program","duration":"Lengd","minutes":"mín","time_remaining":"Tími sem eftir er","no_prediction":"Engin spá","cycle_in_progress":"Hringrás í gangi","status":"Staða","progress":"Framfarir","select_program":"Veldu forrit til að sjá upplýsingar","title":"Titill","status_entity":"Staða eining","icon":"Táknmynd","active_color":"Virkur táknlitur","show_state":"Sýna ástand","show_program":"Sýna dagskrá","show_details":"Sýna upplýsingar","spin_icon":"Snúningstákn (meðan í gangi)","program_entity":"Dagskráreining","pct_entity":"Framvindueining (valfrjálst)","time_entity":"Tímaeining (valfrjálst)","display_mode":"Sýnastilling","show_time_remaining":"Sýna tíma sem eftir er","show_percentage":"Sýna hlutfall","entity_not_found":"Eining fannst ekki","tap_action":"Bankaðu á Aðgerð","hold_action":"Haltu Action","double_tap_action":"Tvíspikkaðu á Action"},"it":{"washer_program":"Programma Lavatrice","program_placeholder":"Seleziona Programma","duration":"Durata","minutes":"min","time_remaining":"Tempo rimanente","no_prediction":"Nessuna previsione","cycle_in_progress":"Ciclo in corso","status":"Stato","progress":"Progressi","select_program":"Seleziona un programma per vedere i dettagli","title":"Titolo","status_entity":"Entità di stato","icon":"Icona","active_color":"Colore icona attiva","show_state":"Mostra stato","show_program":"Mostra programma","show_details":"Mostra dettagli","spin_icon":"Icona che gira (durante la corsa)","program_entity":"Entità del programma","pct_entity":"Entità di avanzamento (facoltativo)","time_entity":"Entità temporale (facoltativo)","display_mode":"Modalità di visualizzazione","show_time_remaining":"Mostra tempo rimanente","show_percentage":"Mostra percentuale","entity_not_found":"Entità non trovata","tap_action":"Tocca Azione","hold_action":"Mantieni Azione","double_tap_action":"Azione doppio tocco"},"ja":{"washer_program":"ウォッシャープログラム","program_placeholder":"プログラムの選択","duration":"間隔","minutes":"分","time_remaining":"残り時間","no_prediction":"予測なし","cycle_in_progress":"進行中のサイクル","status":"状態","progress":"進捗","select_program":"プログラムを選択して詳細を表示します","title":"タイトル","status_entity":"ステータスエンティティ","icon":"アイコン","active_color":"アクティブなアイコンの色","show_state":"状態を表示","show_program":"ショープログラム","show_details":"詳細を表示","spin_icon":"回転アイコン（走行中）","program_entity":"プログラムエンティティ","pct_entity":"進行状況エンティティ (オプション)","time_entity":"時間エンティティ (オプション)","display_mode":"表示モード","show_time_remaining":"残りの上映時間","show_percentage":"パーセンテージを表示","entity_not_found":"エンティティが見つかりません","tap_action":"タップアクション","hold_action":"ホールドアクション","double_tap_action":"ダブルタップアクション"},"ko":{"washer_program":"세탁기 프로그램","program_placeholder":"프로그램 선택","duration":"지속","minutes":"분","time_remaining":"남은 시간","no_prediction":"예측 없음","cycle_in_progress":"사이클 진행 중","status":"상태","progress":"진전","select_program":"세부정보를 보려면 프로그램을 선택하세요.","title":"제목","status_entity":"상태 엔터티","icon":"상","active_color":"활성 아이콘 색상","show_state":"상태 표시","show_program":"쇼 프로그램","show_details":"세부정보 표시","spin_icon":"회전 아이콘(실행 중)","program_entity":"프로그램 엔터티","pct_entity":"진행 엔터티(선택 사항)","time_entity":"시간 엔터티(선택 사항)","display_mode":"디스플레이 모드","show_time_remaining":"남은 시간 표시","show_percentage":"백분율 표시","entity_not_found":"엔터티를 찾을 수 없습니다.","tap_action":"탭 동작","hold_action":"보류 조치","double_tap_action":"더블 탭 액션"},"lt":{"washer_program":"Skalbimo programa","program_placeholder":"Pasirinkite Programa","duration":"Trukmė","minutes":"min","time_remaining":"Likęs laikas","no_prediction":"Jokios prognozės","cycle_in_progress":"Vyksta ciklas","status":"Būsena","progress":"Pažanga","select_program":"Norėdami pamatyti išsamią informaciją, pasirinkite programą","title":"Pavadinimas","status_entity":"Būsenos subjektas","icon":"Piktograma","active_color":"Aktyvios piktogramos spalva","show_state":"Rodyti būseną","show_program":"Rodyti programą","show_details":"Rodyti išsamią informaciją","spin_icon":"Sukimo piktograma (bėgant)","program_entity":"Programos subjektas","pct_entity":"Pažangos subjektas (neprivaloma)","time_entity":"Laiko objektas (neprivaloma)","display_mode":"Ekrano režimas","show_time_remaining":"Rodyti likusį laiką","show_percentage":"Rodyti procentą","entity_not_found":"Subjektas nerastas","tap_action":"Bakstelėkite Veiksmas","hold_action":"Laikyti veiksmą","double_tap_action":"Dukart bakstelėkite veiksmas"},"lv":{"washer_program":"Mazgāšanas programma","program_placeholder":"Atlasiet Programma","duration":"Ilgums","minutes":"min","time_remaining":"Atlikušais laiks","no_prediction":"Nav prognožu","cycle_in_progress":"Notiek cikls","status":"Statuss","progress":"Progress","select_program":"Izvēlieties programmu, lai skatītu detalizētu informāciju","title":"Nosaukums","status_entity":"Statusa entītija","icon":"Ikona","active_color":"Aktīvās ikonas krāsa","show_state":"Rādīt stāvokli","show_program":"Rādīt programmu","show_details":"Rādīt detaļas","spin_icon":"Griešanās ikona (skrienot)","program_entity":"Programmas entītija","pct_entity":"Progresa entītija (neobligāti)","time_entity":"Laika entītija (neobligāti)","display_mode":"Displeja režīms","show_time_remaining":"Rādīt atlikušo laiku","show_percentage":"Rādīt procentus","entity_not_found":"Entītija nav atrasta","tap_action":"Pieskarieties darbībai","hold_action":"Aizturēt darbību","double_tap_action":"Dubultskāriena darbība"},"mk":{"washer_program":"Програма за перење","program_placeholder":"Изберете Програма","duration":"Времетраење","minutes":"мин","time_remaining":"Преостанато време","no_prediction":"Без предвидување","cycle_in_progress":"Циклус во тек","status":"Статус","progress":"Напредок","select_program":"Изберете програма за да видите детали","title":"Наслов","status_entity":"Статусен ентитет","icon":"Икона","active_color":"Активна боја на иконата","show_state":"Прикажи држава","show_program":"Прикажи програма","show_details":"Прикажи детали","spin_icon":"Икона за вртење (додека работи)","program_entity":"Програмски ентитет","pct_entity":"Ентитет за напредок (изборно)","time_entity":"Временски ентитет (изборно)","display_mode":"Режим на прикажување","show_time_remaining":"Прикажи преостанатото време","show_percentage":"Прикажи процент","entity_not_found":"Субјектот не е пронајден","tap_action":"Допрете Акција","hold_action":"Држете акција","double_tap_action":"Акција со двоен допир"},"nb":{"washer_program":"Vaskeprogram","program_placeholder":"Velg Program","duration":"Varighet","minutes":"min","time_remaining":"Gjenstående tid","no_prediction":"Ingen prediksjon","cycle_in_progress":"Syklus pågår","status":"Status","progress":"Framgang","select_program":"Velg et program for å se detaljer","title":"Tittel","status_entity":"Status Entitet","icon":"Ikon","active_color":"Aktiv ikonfarge","show_state":"Vis tilstand","show_program":"Vis program","show_details":"Vis detaljer","spin_icon":"Spinning-ikon (mens du løper)","program_entity":"Program Entitet","pct_entity":"Fremdriftsenhet (valgfritt)","time_entity":"Tidsenhet (valgfritt)","display_mode":"Visningsmodus","show_time_remaining":"Vis gjenværende tid","show_percentage":"Vis prosentandel","entity_not_found":"Enheten ble ikke funnet","tap_action":"Trykk på Handling","hold_action":"Hold handling","double_tap_action":"Dobbelttrykk på handling"},"nl":{"washer_program":"Wasprogramma","program_placeholder":"Selecteer Programma","duration":"Duur","minutes":"min","time_remaining":"Resterende tijd","no_prediction":"Geen voorspelling","cycle_in_progress":"Cyclus in uitvoering","status":"Status","progress":"Voortgang","select_program":"Selecteer een programma om details te bekijken","title":"Titel","status_entity":"Statusentiteit","icon":"Icon","active_color":"Actieve pictogramkleur","show_state":"Toon staat","show_program":"Programma weergeven","show_details":"Details tonen","spin_icon":"Draaiend pictogram (tijdens hardlopen)","program_entity":"Programma-entiteit","pct_entity":"Voortgangsentiteit (optioneel)","time_entity":"Tijdsentiteit (optioneel)","display_mode":"Weergavemodus","show_time_remaining":"Resterende tijd weergeven","show_percentage":"Percentage weergeven","entity_not_found":"Entiteit niet gevonden","tap_action":"Tik op Actie","hold_action":"Actie vasthouden","double_tap_action":"Dubbeltikactie"},"pl":{"washer_program":"Program prania","program_placeholder":"Wybierz Program","duration":"Czas trwania","minutes":"min","time_remaining":"Pozostały czas","no_prediction":"Brak przewidywania","cycle_in_progress":"Cykl w toku","status":"Status","progress":"Postęp","select_program":"Wybierz program, aby zobaczyć szczegóły","title":"Tytuł","status_entity":"Jednostka statusowa","icon":"Ikona","active_color":"Aktywny kolor ikony","show_state":"Pokaż stan","show_program":"Pokaż program","show_details":"Pokaż szczegóły","spin_icon":"Ikona obracania się (podczas biegu)","program_entity":"Jednostka programu","pct_entity":"Jednostka postępu (opcjonalnie)","time_entity":"Jednostka czasu (opcjonalnie)","display_mode":"Tryb wyświetlania","show_time_remaining":"Pokaż pozostały czas","show_percentage":"Pokaż procent","entity_not_found":"Nie znaleziono elementu","tap_action":"Kliknij Akcja","hold_action":"Wstrzymaj akcję","double_tap_action":"Akcja podwójnego dotknięcia"},"pt":{"washer_program":"Programa de lavadora","program_placeholder":"Selecione o programa","duration":"Duração","minutes":"min","time_remaining":"Tempo restante","no_prediction":"Sem previsão","cycle_in_progress":"Ciclo em andamento","status":"Status","progress":"Progresso","select_program":"Selecione um programa para ver detalhes","title":"Título","status_entity":"Entidade de status","icon":"Ícone","active_color":"Cor do ícone ativo","show_state":"Mostrar estado","show_program":"Mostrar programa","show_details":"Mostrar detalhes","spin_icon":"Ícone giratório (durante a execução)","program_entity":"Entidade do Programa","pct_entity":"Entidade de progresso (opcional)","time_entity":"Entidade de tempo (opcional)","display_mode":"Modo de exibição","show_time_remaining":"Mostrar tempo restante","show_percentage":"Mostrar porcentagem","entity_not_found":"Entidade não encontrada","tap_action":"Toque em Ação","hold_action":"Manter ação","double_tap_action":"Ação de toque duplo"},"pt-BR":{"washer_program":"Programa de lavadora","program_placeholder":"Selecione o programa","duration":"Duração","minutes":"min","time_remaining":"Tempo restante","no_prediction":"Sem previsão","cycle_in_progress":"Ciclo em andamento","status":"Status","progress":"Progresso","select_program":"Selecione um programa para ver detalhes","title":"Título","status_entity":"Entidade de status","icon":"Ícone","active_color":"Cor do ícone ativo","show_state":"Mostrar estado","show_program":"Mostrar programa","show_details":"Mostrar detalhes","spin_icon":"Ícone giratório (durante a execução)","program_entity":"Entidade do Programa","pct_entity":"Entidade de progresso (opcional)","time_entity":"Entidade de tempo (opcional)","display_mode":"Modo de exibição","show_time_remaining":"Mostrar tempo restante","show_percentage":"Mostrar porcentagem","entity_not_found":"Entidade não encontrada","tap_action":"Toque em Ação","hold_action":"Manter ação","double_tap_action":"Ação de toque duplo"},"ro":{"washer_program":"Program de spălat","program_placeholder":"Selectați Program","duration":"Durată","minutes":"min","time_remaining":"Timp rămas","no_prediction":"Fără predicție","cycle_in_progress":"Ciclu în curs","status":"Stare","progress":"Progres","select_program":"Selectați un program pentru a vedea detalii","title":"Titlu","status_entity":"Entitate de stare","icon":"Pictogramă","active_color":"Culoarea pictogramei active","show_state":"Arată stare","show_program":"Arată programul","show_details":"Afișați detalii","spin_icon":"Pictogramă care se învârte (în timpul alergării)","program_entity":"Entitatea de program","pct_entity":"Entitate de progres (opțional)","time_entity":"Entitate oră (Opțional)","display_mode":"Modul de afișare","show_time_remaining":"Arată timpul rămas","show_percentage":"Arată procentul","entity_not_found":"Entitatea nu a fost găsită","tap_action":"Atingeți Acțiune","hold_action":"Țineți Acțiune","double_tap_action":"Atingeți de două ori Acțiune"},"ru":{"washer_program":"Программа стирки","program_placeholder":"Выберите программу","duration":"Продолжительность","minutes":"мин","time_remaining":"Оставшееся время","no_prediction":"Нет прогноза","cycle_in_progress":"Цикл в процессе","status":"Статус","progress":"Прогресс","select_program":"Выберите программу, чтобы увидеть подробности","title":"Заголовок","status_entity":"Статус объекта","icon":"Икона","active_color":"Цвет активного значка","show_state":"Показать состояние","show_program":"Шоу-программа","show_details":"Показать детали","spin_icon":"Вращающийся значок (во время бега)","program_entity":"Программный объект","pct_entity":"Сущность прогресса (необязательно)","time_entity":"Сущность времени (необязательно)","display_mode":"Режим отображения","show_time_remaining":"Показать оставшееся время","show_percentage":"Показать процент","entity_not_found":"Объект не найден","tap_action":"Нажмите «Действие».","hold_action":"Удерживать действие","double_tap_action":"Двойное нажатие"},"sk":{"washer_program":"Program práčky","program_placeholder":"Vyberte položku Program","duration":"Trvanie","minutes":"min","time_remaining":"Zostávajúci čas","no_prediction":"Žiadna predpoveď","cycle_in_progress":"Prebiehajúci cyklus","status":"Stav","progress":"Pokrok","select_program":"Ak chcete zobraziť podrobnosti, vyberte program","title":"Názov","status_entity":"Stavová entita","icon":"Ikona","active_color":"Farba aktívnej ikony","show_state":"Zobraziť stav","show_program":"Zobraziť program","show_details":"Zobraziť podrobnosti","spin_icon":"Ikona otáčania (pri behu)","program_entity":"Programová entita","pct_entity":"Entita pokroku (voliteľné)","time_entity":"Časová entita (voliteľné)","display_mode":"Režim zobrazenia","show_time_remaining":"Zobraziť zostávajúci čas","show_percentage":"Zobraziť percento","entity_not_found":"Entita sa nenašla","tap_action":"Klepnite na položku Akcia","hold_action":"Hold Action","double_tap_action":"Akcia dvojitého klepnutia"},"sl":{"washer_program":"Program za pranje","program_placeholder":"Izberite Program","duration":"Trajanje","minutes":"min","time_remaining":"Preostali čas","no_prediction":"Brez napovedi","cycle_in_progress":"Cikel v teku","status":"Stanje","progress":"Napredek","select_program":"Za ogled podrobnosti izberite program","title":"Naslov","status_entity":"Statusna entiteta","icon":"Ikona","active_color":"Barva aktivne ikone","show_state":"Prikaži stanje","show_program":"Show Program","show_details":"Pokaži podrobnosti","spin_icon":"Vrteča se ikona (med tekom)","program_entity":"Programska entiteta","pct_entity":"Entiteta napredka (neobvezno)","time_entity":"Časovna entiteta (neobvezno)","display_mode":"Način prikaza","show_time_remaining":"Prikaži preostali čas","show_percentage":"Pokaži odstotek","entity_not_found":"Entiteta ni najdena","tap_action":"Tapnite Dejanje","hold_action":"Zadrži akcijo","double_tap_action":"Dejanje dvojnega dotika"},"sq":{"washer_program":"Programi i larës","program_placeholder":"Zgjidhni Programin","duration":"Kohëzgjatja","minutes":"min","time_remaining":"Koha e mbetur","no_prediction":"Asnjë Parashikim","cycle_in_progress":"Cikli në vazhdim","status":"Statusi","progress":"Përparim","select_program":"Zgjidhni një program për të parë detajet","title":"Titulli","status_entity":"Entiteti i statusit","icon":"Ikona","active_color":"Ngjyra e ikonës aktive","show_state":"Trego shtetin","show_program":"Shfaq programin","show_details":"Shfaq Detajet","spin_icon":"Ikona rrotulluese (Gjatë funksionimit)","program_entity":"Subjekti i programit","pct_entity":"Entiteti i progresit (opsionale)","time_entity":"Entiteti i kohës (opsionale)","display_mode":"Modaliteti i shfaqjes","show_time_remaining":"Shfaq kohën e mbetur","show_percentage":"Shfaq përqindjen","entity_not_found":"Subjekti nuk u gjet","tap_action":"Prekni Veprim","hold_action":"Mbajeni veprimin","double_tap_action":"Veprimi i prekjes së dyfishtë"},"sr-Latn":{"washer_program":"Program pranja","program_placeholder":"Izaberi program","duration":"Trajanje","minutes":"min","time_remaining":"Preostalo vreme","no_prediction":"Nema procene","cycle_in_progress":"Ciklus je u toku","status":"Status","progress":"Napredak","select_program":"Izaberite program da vidite detalje","title":"Naslov","status_entity":"Entitet statusa","icon":"Ikona","active_color":"Boja aktivne ikone","show_state":"Prikaži stanje","show_program":"Prikaži program","show_details":"Prikaži detalje","spin_icon":"Rotirajuća ikona (dok radi)","program_entity":"Entitet programa","pct_entity":"Entitet napretka (opciono)","time_entity":"Entitet vremena (opciono)","display_mode":"Režim prikaza","show_time_remaining":"Prikaži preostalo vreme","show_percentage":"Prikaži procenat","entity_not_found":"Entitet nije pronađen","tap_action":"Radnja na dodir","hold_action":"Radnja na zadržavanje","double_tap_action":"Radnja na dvostruki dodir"},"sv":{"washer_program":"Tvättprogram","program_placeholder":"Välj Program","duration":"Varaktighet","minutes":"min","time_remaining":"Återstående tid","no_prediction":"Ingen förutsägelse","cycle_in_progress":"Cykel pågår","status":"Status","progress":"Framsteg","select_program":"Välj ett program för att se detaljer","title":"Titel","status_entity":"Status Entitet","icon":"Ikon","active_color":"Aktiv ikonfärg","show_state":"Visa tillstånd","show_program":"Visa program","show_details":"Visa detaljer","spin_icon":"Spinning-ikon (medan du springer)","program_entity":"Program Entitet","pct_entity":"Progress Entity (valfritt)","time_entity":"Tidsenhet (valfritt)","display_mode":"Visningsläge","show_time_remaining":"Visa återstående tid","show_percentage":"Visa procent","entity_not_found":"Enheten hittades inte","tap_action":"Tryck på Åtgärd","hold_action":"Håll Action","double_tap_action":"Dubbeltrycksåtgärd"},"tr":{"washer_program":"Yıkama Programı","program_placeholder":"Program Seç","duration":"Süre","minutes":"dk.","time_remaining":"Kalan Süre","no_prediction":"Tahmin Yok","cycle_in_progress":"Döngü devam ediyor","status":"Durum","progress":"İlerlemek","select_program":"Ayrıntıları görmek için bir program seçin","title":"Başlık","status_entity":"Durum Varlığı","icon":"Simge","active_color":"Etkin Simge Rengi","show_state":"Durumu Göster","show_program":"Programı Göster","show_details":"Ayrıntıları Göster","spin_icon":"Dönen Simge (Koşarken)","program_entity":"Program Varlığı","pct_entity":"İlerleme Varlığı (İsteğe bağlı)","time_entity":"Zaman Varlığı (İsteğe Bağlı)","display_mode":"Ekran Modu","show_time_remaining":"Kalan Süreyi Göster","show_percentage":"Yüzdeyi Göster","entity_not_found":"Varlık bulunamadı","tap_action":"Eylem'e dokunun","hold_action":"Eylemi Beklet","double_tap_action":"Çift Dokunma Eylemi"},"uk":{"washer_program":"Програма прання","program_placeholder":"Виберіть програму","duration":"Тривалість","minutes":"хв","time_remaining":"Час, що залишився","no_prediction":"Без передбачення","cycle_in_progress":"Цикл триває","status":"Статус","progress":"Прогрес","select_program":"Виберіть програму, щоб переглянути деталі","title":"Назва","status_entity":"Status Entity","icon":"значок","active_color":"Активний колір значка","show_state":"Показати стан","show_program":"Шоу програма","show_details":"Показати деталі","spin_icon":"Піктограма обертання (під час бігу)","program_entity":"Програмна сутність","pct_entity":"Сутність прогресу (необов'язково)","time_entity":"Сутність часу (необов’язково)","display_mode":"Режим відображення","show_time_remaining":"Показати час, що залишився","show_percentage":"Показати відсоток","entity_not_found":"Об'єкт не знайдено","tap_action":"Натисніть Дія","hold_action":"Дія утримання","double_tap_action":"Подвійне торкання"},"zh-Hans":{"washer_program":"清洗程序","program_placeholder":"选择节目","duration":"期间","minutes":"分钟","time_remaining":"剩余时间","no_prediction":"没有预测","cycle_in_progress":"循环正在进行中","status":"地位","progress":"进步","select_program":"选择一个程序以查看详细信息","title":"标题","status_entity":"状态实体","icon":"图标","active_color":"活动图标颜色","show_state":"显示状态","show_program":"演出节目","show_details":"显示详情","spin_icon":"旋转图标（运行时）","program_entity":"程序实体","pct_entity":"进度实体（可选）","time_entity":"时间实体（可选）","display_mode":"显示模式","show_time_remaining":"显示剩余时间","show_percentage":"显示百分比","entity_not_found":"未找到实体","tap_action":"点击操作","hold_action":"保持行动","double_tap_action":"双击操作"}};

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
      spin_icon: true,
      tap_action: { action: "more-info" },
      hold_action: { action: "none" },
      double_tap_action: { action: "none" }
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
    // Gesture state for tap / hold / double-tap recognition.
    this._holdTimer = null;
    this._holdTriggered = false;
    this._tapTimer = null;
    this._lastTapTime = 0;
    this._pointerStart = null;
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onPointerCancel = this._onPointerCancel.bind(this);
  }

  disconnectedCallback() {
    // Avoid stray actions firing after the card is removed from the DOM.
    this._clearHoldTimer();
    if (this._tapTimer) {
      window.clearTimeout(this._tapTimer);
      this._tapTimer = null;
    }
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

  _clearHoldTimer() {
    if (this._holdTimer) {
      window.clearTimeout(this._holdTimer);
      this._holdTimer = null;
    }
  }

  _onPointerDown(ev) {
    // Only react to the primary pointer (left mouse button / touch / pen).
    if (ev.button !== undefined && ev.button !== 0) return;
    this._holdTriggered = false;
    this._pointerCanceled = false;
    this._pointerStart = { x: ev.clientX, y: ev.clientY };

    const holdCfg = this._cfg && this._cfg.hold_action;
    if (holdCfg && holdCfg.action && holdCfg.action !== "none") {
      this._clearHoldTimer();
      this._holdTimer = window.setTimeout(() => {
        this._holdTimer = null;
        this._holdTriggered = true;
        this._fireHaptic("success");
        this._executeAction(holdCfg);
      }, HOLD_MS);
    }
  }

  _onPointerMove(ev) {
    // Cancel the gesture if the pointer drifts (e.g. the user is scrolling), so
    // neither the pending hold nor the release-tap fires.
    if (!this._pointerStart) return;
    const dx = ev.clientX - this._pointerStart.x;
    const dy = ev.clientY - this._pointerStart.y;
    if (dx * dx + dy * dy > TAP_MOVE_TOLERANCE * TAP_MOVE_TOLERANCE) {
      this._clearHoldTimer();
      this._pointerStart = null;
      this._pointerCanceled = true;
    }
  }

  _onPointerCancel() {
    this._clearHoldTimer();
  }

  _onPointerUp() {
    this._clearHoldTimer();
    // The pointer drifted (scroll/drag): the release should not count as a tap.
    if (this._pointerCanceled) {
      this._pointerCanceled = false;
      return;
    }
    // A hold already fired its action; the release should not also count as a tap.
    if (this._holdTriggered) {
      this._holdTriggered = false;
      return;
    }

    const tapCfg = (this._cfg && this._cfg.tap_action) || { action: "more-info" };
    const doubleCfg = this._cfg && this._cfg.double_tap_action;
    const hasDouble = doubleCfg && doubleCfg.action && doubleCfg.action !== "none";

    // With no double-tap action configured, fire the tap immediately (no latency).
    if (!hasDouble) {
      this._executeAction(tapCfg);
      return;
    }

    const now = Date.now();
    if (this._tapTimer && now - this._lastTapTime < DOUBLE_TAP_MS) {
      window.clearTimeout(this._tapTimer);
      this._tapTimer = null;
      this._lastTapTime = 0;
      this._executeAction(doubleCfg);
      return;
    }

    // First tap: wait briefly to see whether a second one arrives.
    this._lastTapTime = now;
    this._tapTimer = window.setTimeout(() => {
      this._tapTimer = null;
      this._executeAction(tapCfg);
    }, DOUBLE_TAP_MS);
  }

  _fireHaptic(type) {
    this.dispatchEvent(new CustomEvent("haptic", {
      detail: type,
      bubbles: true,
      composed: true,
    }));
  }

  _executeAction(actionCfg) {
    if (!actionCfg) return;
    const action = actionCfg.action || "more-info";
    const entityId = actionCfg.entity || (this._cfg && this._cfg.entity);

    switch (action) {
      case "none":
        return;

      case "more-info": {
        if (!entityId) return;
        this.dispatchEvent(new CustomEvent("hass-more-info", {
          detail: { entityId },
          bubbles: true,
          composed: true,
        }));
        return;
      }

      case "toggle": {
        // homeassistant.toggle routes to the correct domain service for all
        // common toggleable domains, so no per-domain table is needed.
        if (!this._hass || !entityId) return;
        this._hass.callService("homeassistant", "toggle", { entity_id: entityId });
        return;
      }

      case "call-service":
      case "perform-action": {
        const svc = actionCfg.perform_action || actionCfg.service;
        if (!svc || !this._hass) return;
        const [svcDomain, svcName] = svc.split(".");
        if (!svcDomain || !svcName) return;
        const data = { ...(actionCfg.data || actionCfg.service_data || {}) };
        this._hass.callService(svcDomain, svcName, data, actionCfg.target);
        return;
      }

      case "navigate": {
        const path = actionCfg.navigation_path;
        if (!path) return;
        if (actionCfg.navigation_replace) {
          window.history.replaceState(window.history.state, "", path);
        } else {
          window.history.pushState(null, "", path);
        }
        window.dispatchEvent(new CustomEvent("location-changed", {
          detail: { replace: !!actionCfg.navigation_replace },
        }));
        return;
      }

      case "url": {
        const url = actionCfg.url_path;
        if (!url) return;
        // noopener/noreferrer prevents the opened page from reaching back via
        // window.opener (reverse tabnabbing).
        window.open(url, "_blank", "noopener,noreferrer");
        return;
      }

      default:
        // Unsupported actions (e.g. "assist") are intentionally ignored: a
        // standalone card resource cannot resolve Home Assistant's internal
        // dialog chunks, so attempting them would only throw at runtime.
        return;
    }
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

      const cardEl = this.shadowRoot.getElementById("card");
      cardEl.addEventListener("pointerdown", this._onPointerDown);
      cardEl.addEventListener("pointermove", this._onPointerMove);
      cardEl.addEventListener("pointerup", this._onPointerUp);
      cardEl.addEventListener("pointercancel", this._onPointerCancel);
      cardEl.addEventListener("pointerleave", this._onPointerCancel);
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

    const title = this._cfg.title || "Washing Machine";
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
        { name: "tap_action", selector: { ui_action: {} } },
        { name: "hold_action", selector: { ui_action: {} } },
        { name: "double_tap_action", selector: { ui_action: {} } },
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
          display_mode: this._getTranslation("display_mode"),
          tap_action: this._getTranslation("tap_action"),
          hold_action: this._getTranslation("hold_action"),
          double_tap_action: this._getTranslation("double_tap_action")
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
