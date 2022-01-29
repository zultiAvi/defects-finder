# Toy project: Defects finder
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for examples](https://static.wixstatic.com/media/d147d7_f0c64bd94dda45e7b0203104cbe3f69a~mv2.png)

- [תיאור](#תיאור)
- [תהליך העבודה](#תהליך-העבודה)
  - [רעיונות](#רעיונות)
  - [הנחות עבודה](#הנחות-עבודה)
  - [הפתרון הנבחר](#הפתרון-הנבחר)
  - [תוצאות](#תוצאות)
  - [שיפורים אפשריים](#שיפורים-אפשריים)
- [שימוש](#שימוש)
  - [התקנה](#התקנה)
  - [ייצור דאטה](#ייצור-דאטה)
  - [אימון](#אימון)
  - [חיזוי](#חיזוי)


## תיאור

חבר איתגר אותי בעבודת בית שניתנת לסטודנטים, המטרה למצוא דפקטים בתהליך ייצור של שבבים. אני קיבלתי 3 תמונות בלבד ומהן יש להכין אלגוריתם המזהה את הפגמים בייצור, גם לפגמים מסוגים שונים שלא ראיתי.
לקחתי על עצמי את האתגר ומספר ימים לאחר מכן אני שולח לו את הגיט הזה.
מקוה שתמצאו עניין בתהליך העבודה, ברעיונות שעלו ובקוד.
את הרשת שהשתמשתי בה לקחתי מ github של [github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

תודה

אבי.

נ.ב. התמונות למעלה הן ספויילר לתוצאות העבודה

---
## תהליך העבודה

### רעיונות
כאשר קיבלתי את הפרוייקט מספר רעיונות לפתרון עלו: 
1. פתרון בשיטות קלאסיות הכולל:
    1. מציאת ההזזה המתאימה בין שתי התמונות הרלוונטיות
    2. הפרש בין התמונות 
    3. מציאת האזורים השונים מ 0 וסימונם
2. ML model
    1. רשת המזהה הבדלים בין שתי תמונות שכבר aligned
    2. רשת המסתכלת על קטע קטן של התמונה ומחפשת בו את השגיאות

### הנחות עבודה
מספר הנחות עבודה:
1. התעלמתי מזמן ריצת הפתרון (!) מכיוון שזהו בעקרון POC, אשתדל להראות תוצאות זיהוי טובות ואציג מספר נקודות בהן ניתן לשפר ביצועים במידת הצורך.
2. התמונה מכילה 3 "צבעים" עיקריים: אחד עבור פנים השבב, השני עבור הרווחים והשלישי עבור הגבולות.
3. קווי הגבול הינם אופקיים ואנכיים בתוספת של חצאי עיגולים בקצות הקטע. 
4. ההזזה בין שתי תמונות הינה קטנה יחסית (עד 30 פיקסלים בכל כיוון)
5. הפגמים הינם קטנים יחסית (עד 32 פיקסלים) - כאשר הפגמים גדולים מאד יתכנו בעיות בשני הפתרונות


התחלתי לממש את הפתרון בשיטה הקלאסית. מציאת ההזזה המתאימה בין התמונות והפרש בין התמונות המוזזות.
אולם מכיוון שהתמונות שונות גם באזורים בהן אין פגמים ואפילו שההזזה מושלמת, לא הצלחתי למצוא דרך מספיק רובסטית לפתרון ועברתי לשיטת ה ML.

### הפתרון הנבחר
* הפתרון מקבל שתי תמונות (inspected ו reference) ומחזיר תמונת גילוי
* הפתרון מתחיל במציאת ההזזה בין התמונות וחיתוך התמונות כך שישבו בדיוק אחת על השנייה (תתכן בכל זאת הזזה של מספר פיקסלים בין התמונות)
* מכיוון שאני מניח  שגודל הפגם הינו קטן, אני מייצר רשת המקבלת ריבוע של 64*64 פיקסלים משתי התמונות ומחפשת פגמים בתמונה הראשונה כאשר השניה משמשת כבקרה
* מכיוון שאני קיבלתי מעט מאד Data (שתי תמונות המכילות 3 פגמים כל אחת), לקחתי כל תמונה וייצרתי ממנה הרבה ריבועים של 64*64 ולאחר מכן הוספתי באופן מלאכותי פגמים לתמונה.
* אני מקווה שהפגמים שהוספו באופן מלאכותי מתאימים לפגמים האמיתיים שקיימים בשטח. כמובן שככל שיהיו יותר פגמים וסוגי שבבים ה data שייכנס לרשת יהיה אייכותי יותר וסיכויי ההצלחה של הרשת יגדלו משמעותית.
* יצירת הפגמים בוצעה באופן הבא:
    * ייוצרו באופן מלאכותי כ 20 סוגי פגמים שונים (מצורפים בספרייות data/defect_masks)
    * לכל תמונה נבחר באקראי אחד מהפגמים. פגם זה סובב והוזז באופן אקראי בתוך התמונה.
    * בתוך אזור זה שונו גוני האפור מהתמונה המקורית ב 15-65 גוונים
    * תמונת ה referrence הוזזה ב +/- 3 פיקסלים.
    * בתמונת ה referrence ובתמונת ה inspection שונו גוני האפור של כל הפיקסלים ב+/- 7 גוונים.
* השתמשתי ברשת U-net שנכתבה ב PyTorch. את הרשת הורדתי מ github. ניתן להגיע אליו מהלינק - [github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) ואני חושב שכמעט ולא שיניתי שום דבר מהותי.
* את הרשת אימנתי על המחשב בבית,  לקח כשעה לאמן 5 אפוקים עבור כ 10,000 תמונות. ה dice coeff הגיע ל כ 0.65 (לבסוף המשכתי את אימון הרשת כ 5 פעמים בכל פעם עם דאטה סט אחר)
* תוצאות הרשת היו משביעות רצון והרשת זיהתה בצורה טובה את הבעיות שייצרתי.
* על מנת לא ליצור bias עבור הבעיות הקיימות בדוגמאות, לא הכנסתי לdata set אף אחד מהפגמים המקוריים.
* הרשת ביצעה פרדיקציה לכ 70 תמונות בשניה על גבי ה cpu/gpu.
* מספר דוגמאות מדאטה סט של האימון ופרדיקציות הרשת עבורן:
* ![samples from training dataset](https://static.wixstatic.com/media/d147d7_037cc58c248b4f93ba8a54caea5c512f~mv2.png)
* עבור פרדיקציה לתמונה גדולה:
    1. מציאת ההזזה בין תמונת ה inspect לתמונת ה referrence.
    2. פירוק התמונה לתמונות של 64*64 עם חפיפה של 32 פיקסלים בכל כיוון.
    3. הרצת הרשת והחזרת התוצאות
    4. לבסוף, חיבור התוצאות לתמונת חיזוי אחת. 

### תוצאות
קיבלתי את התוצאות הבאות:

* כל התמונות והתוצאות נמצאים בספריית data/results
* עבור שלושת התמונות שסופקו (2 עם בעיות ואחת בלי בעיות):
  * המודל סימן את כל הבעיות במקומות הנכונים
  * גודל הסימון היה טוב וקרוב מאד לגודל המדוייק
  * פונקציית הפרדיקציה לקחה בערך 2 שניות
* תמונות החיזוי מופיעות בתחילת הדף.
* שימו לב שכל תמונה שהכילה בעיה בתמונה הראשית לא נכנסה לדאטה לאימון
* מכיוון שאני מחלק את התמונה הגדולה להרבה תמונות קטנות, ניתן לראות שלא קיבלנו False Positive עבור דוגמאות אלו

### שיפורים אפשריים
שינויים ושיפורים נוספים בפרוייקט:
1. קבלת עוד data (בעיקר אמיתי אם קיים)
2. שיפור סימולציית השגיאות
3. החלפת המודל במודל אחר.
4. לבצע אופטימיזציות על המודל עצמו, שינוי loss, learning rate, ...
5. שימוש בתמונת ההפרשים בין התמונות על מנת לזהות מקומות חשודים ולהריץ את הרשת רק שם, לשיפור ביצועים.
6. הקטנת המודל על מנת לשפר ביצועים מבלי לפגוע באיכות הזיהוי
7. עבודה בשיטת in-painting לזיהוי הפגמים:
    *  מאמנים רשת שמסוגלת להשלים את התמונה אם מוחקים חלר ממנה.
    *  מזהים אזור חשוד ומוחקים אותו.
    *  נותנים לרשת להשלים אזור זה
    *  משווים את השלמת הרשת לתמונה האמיתית


תודה

אבי

----

## שימוש

### התקנה

1. Please read  [**milesial-Pytorch-UNet-README.md**](milesial-Pytorch-UNet-README.md) for full installation instructions.
2. Download model from [model.pth](https://drive.google.com/file/d/1fv5T_TNhCR1ppvgZcgpx32k4bHGvvMh4) to ./models/model.pth


### ייצור דאטה
```console
> python ./data_generation.py --input {cases for generation (spaces between them)} 
usage: data_generation.py [-h] --input INPUT [INPUT ...]
                          [--input-folder in_folder]
                          [--output-folder out_folder]
                          [--masks-folder masks_folder] [--mask-size size]
                          [--seed SEED] [--step STEP]
                          [--low-intensity LOW_INTENSITY]
                          [--high-intensity HIGH_INTENSITY]

generate training data from input images

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        cases names (file names will be {case} +
                        "_inspecetd_image.tif" and
                        {case}+"_reference_image.tif"
  --input-folder in_folder, -if in_folder
                        Input folder
  --output-folder out_folder, -o out_folder
                        Output folder (images will be created in {folder} /
                        "imgs" and {folder} / "masks"
  --masks-folder masks_folder, -mf masks_folder
                        Defect masks folder for GT modification
  --mask-size size, -ms size
                        Size of mask
  --seed SEED, -s SEED  Seed for the random (-1 to use time)
  --step STEP, -st STEP
                        Step difference between two images taken
  --low-intensity LOW_INTENSITY, -li LOW_INTENSITY
                        Min intensity of error
  --high-intensity HIGH_INTENSITY, -hi HIGH_INTENSITY
                        Max intensity of error
```

### אימון
```console
> python ./train.py 
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

### חיזוי
```console
> python ./predict_big_image.py --model {model.pth} --input {cases for prediction (spaces between them)}
usage: predict_big_image.py [-h] [--model FILE] --input INPUT [INPUT ...]
                            [--input-folder in_folder] [--output-folder out_folder]
                            [--mask-size size] [--viz] [--no-save]
                            [--mask-threshold MASK_THRESHOLD]
                            [--debug-folder DEBUG_FOLDER]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        cases names (file names will be {case} +
                        "_inspecetd_image.tif" and
                        {case}+"_reference_image.tif"
  --input-folder in_folder, -if in_folder
                        Input folder for images
  --output-folder out_folder, -of out_folder
                        Output folder for the result detection mask
  --mask-size size, -ms size
                        Size of mask
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white
  --debug-folder DEBUG_FOLDER, -d DEBUG_FOLDER
                        Debug folder, to write temporary small images.
```
    
----
