import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
import random
import arabic_reshaper
from bidi.algorithm import get_display
from gtts import gTTS
import pygame
import io
import threading
from PIL import Image, ImageDraw, ImageFont
import os
#from helpers import FingerStabilizer, ProgressiveNumberChallenge, MonthQuizChallenge


# Function to properly display Arabic text
def display_arabic(text):
    """Convert Arabic text for proper display"""
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except:
        return text

# Function to put Arabic text on OpenCV image using PIL
def put_arabic_text_opencv(img, text, position, font_size=40, color=(255, 255, 255)):
    """Put Arabic text on OpenCV image using PIL for proper Arabic rendering"""
    try:
        # Convert BGR to RGB for PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Process Arabic text for proper display
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        
        # Try to load Arabic font, fallback to default
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf", 
            "C:/Windows/Fonts/calibri.ttf",
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        ]
        
        font = None
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Draw text on PIL image
        draw.text(position, bidi_text, font=font, fill=color)
        
        # Convert back to BGR for OpenCV
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"Error rendering Arabic text: {e}")
        # Fallback to simple text
        cv2.putText(img, "Arabic Text Error", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# Function to convert number to Arabic words
def number_to_arabic_words(number):
    """Convert number to Arabic pronunciation"""
    ones = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]
    teens = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", 
             "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
    tens = ["", "", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"]
    hundreds = ["", "مئة", "مئتان", "ثلاثمئة", "أربعمئة", "خمسمئة", "ستمئة", "سبعمئة", "ثمانمئة", "تسعمئة"]
    
    if number == 0:
        return "صفر"
    
    result = []
    
    # Ten thousands
    if number >= 10000:
        ten_thousands = number // 10000
        if ten_thousands == 1:
            result.append("عشرة آلاف")
        else:
            result.append(ones[ten_thousands] + " عشرة آلاف")
        number %= 10000
    
    # Thousands
    if number >= 1000:
        thousands = number // 1000
        if thousands == 1:
            result.append("ألف")
        elif thousands == 2:
            result.append("ألفان")
        elif thousands < 10:
            result.append(ones[thousands] + " آلاف")
        else:
            result.append(number_to_arabic_words(thousands) + " ألف")
        number %= 1000
    
    # Hundreds
    if number >= 100:
        hundred = number // 100
        result.append(hundreds[hundred])
        number %= 100
    
    # Tens and ones
    if number >= 20:
        ten = number // 10
        result.append(tens[ten])
        number %= 10
        if number > 0:
            result.append(ones[number])
    elif number >= 10:
        result.append(teens[number - 10])
    elif number > 0:
        result.append(ones[number])
    
    return " ".join(result)

# Function to speak Arabic text using gTTS
def speak_arabic(text, lang='ar'):
    """Generate and play Arabic speech"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_fp)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in TTS: {e}")

# Function to load model per level
def load_model_for_level(level):
    if level == 1:
        return YOLO('best4.pt')
    return None

# Function to find available camera
def find_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()
    return None

# Count extended fingers
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    count = 0

    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
        count += 1

    for i in range(1, 5):
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
            count += 1

    return count

# Initialize
level = 1
model = load_model_for_level(level)

# Initialize MediaPipe with improved settings for level 3
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create two different hand detection instances
hands_general = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=0
)

# More sensitive hand detection for level 3
hands_level3 = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Arabic month names
months = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جمادى الأولى", "جمادى الآخرة",
    "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة"
]

# Object names in Arabic (you can expand this dictionary)
object_names_arabic = {
    "person": "شخص",
    "car": "سيارة", 
    "bus": "حافلة",
    "truck": "شاحنة",
    "bike": "دراجة",
    "phone": "هاتف",
    "laptop": "حاسوب محمول",
    "mouse": "فأرة",
    "keyboard": "لوحة مفاتيح",
    "book": "كتاب",
    "cup": "كوب",
    "bottle": "زجاجة",
    "chair": "كرسي",
    "table": "طاولة",
    "tv": "تلفزيون",
    "clock": "ساعة",
    "apple": "تفاحة",
    "banana": "موزة",
    "orange": "برتقال",
    "dog": "كلب",
    "cat": "قطة",
    "bird": "طائر"
}

# Level 2 Number Challenge Variables
number_challenge_active = False
target_number = 0
user_input_digits = []
current_digit_position = 0
digit_capture_time = 0
last_finger_count = 0
stable_finger_time = 0
number_challenge_complete = False

cap = find_camera()
if cap is None:
    print(display_arabic("لم يتم العثور على كاميرا متصلة."))
    exit()

last_state = {'object': None, 'hand': None}
pause_finger_until = 0
quiz_mode = False
quiz_month = ""
quiz_answered = False
quiz_start_time = 0
first_gesture_time = 0
first_gesture_count = 0

# Initialize pygame mixer for TTS
try:
    pygame.mixer.init()
except:
    print("Warning: Could not initialize pygame mixer for TTS")

# Print instructions in Arabic
print("=" * 50)
print(display_arabic("برنامج كشف الكائنات والأيدي"))
print("=" * 50)
print(display_arabic("التحكم:"))
print(display_arabic("• اضغط 'w' للمستوى الثاني - تحدي الأرقام"))
print(display_arabic("• اضغط 'r' للمستوى الثالث - اختبار الشهور"))
print(display_arabic("• اضغط مسطرة المسافة للعودة للمستوى الأول"))
print(display_arabic("• اضغط 'n' لرقم جديد في المستوى الثاني"))
print(display_arabic("• اضغط 'q' للخروج"))
print("=" * 50)
print(display_arabic("المستوى الأول - كشف الكائنات"))
print(display_arabic("الكاميرا مفتوحة..."))

def start_number_challenge():
    """Start a new number challenge"""
    global target_number, user_input_digits, current_digit_position
    global digit_capture_time, number_challenge_active, number_challenge_complete
    
    target_number = random.randint(0, 99999)
    user_input_digits = []
    current_digit_position = 0
    digit_capture_time = 0
    number_challenge_active = True
    number_challenge_complete = False
    
    # Convert number to Arabic and speak it
    arabic_text = number_to_arabic_words(target_number)
    print(f"\n{display_arabic('رقم جديد:')} {target_number}")
    print(f"{display_arabic('بالعربية:')} {arabic_text}")
    print(f"{display_arabic('أظهر كل رقم بأصابعك بالترتيب')}")
    
    # Speak the number in a separate thread to avoid blocking
    def speak_thread():
        speak_arabic(arabic_text)
    
    threading.Thread(target=speak_thread, daemon=True).start()

def new_quiz():
    global quiz_mode, quiz_answered, quiz_month, quiz_start_time, first_gesture_time, first_gesture_count

    time.sleep(3)
    quiz_mode = True
    quiz_answered = False
    quiz_month = random.choice(months)
    quiz_start_time = time.time()
    first_gesture_time = 0
    first_gesture_count = 0

    # Process month name separately first
    shaped_month = display_arabic(quiz_month)

    # Now combine it with the rest of the text
    full_text = display_arabic(f"سؤال جديد: الشهر هو {shaped_month} - أظهر رقم هذا الشهر بأصابعك")
    print(full_text)

    if months.index(quiz_month) + 1 > 10:
        print(display_arabic("للأشهر 11 و 12: أظهر الرقم الأول ثم الرقم الثاني خلال 7 ثوان"))

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        print(display_arabic("فشل في قراءة الكاميرا. تأكد من الاتصال."))
        break

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        level = 2
        print(f"\n{display_arabic('المستوى الثاني - تحدي الأرقام')}")
        start_number_challenge()
        last_state = {'object': None, 'hand': None}
    elif key == ord('n') and level == 2:
        start_number_challenge()
    elif key == ord('r'):
        level = 3
        quiz_mode = True
        quiz_answered = False
        quiz_month = random.choice(months)
        quiz_start_time = time.time()
        first_gesture_time = 0
        first_gesture_count = 0
        print(f"\n{display_arabic('المستوى الثالث - اختبار الشهور')}")
        print(f"{display_arabic('السؤال: الشهر هو')} '{quiz_month}' - {display_arabic('أظهر رقم هذا الشهر بأصابعك')}")
        if months.index(quiz_month) + 1 > 10:
            print(f"{display_arabic('للأشهر 11 و 12: أظهر الرقم الأول ثم الرقم الثاني خلال 7 ثوان')}")
    elif key == ord(' '):
        level = 1
        model = load_model_for_level(level)
        if model is None:
            model = YOLO('best4.pt')
        print(f"\n{display_arabic('المستوى الأول - كشف الكائنات')}")
        number_challenge_active = False
        last_state = {'object': None, 'hand': None}
    elif key == ord('q'):
        print(f"\n{display_arabic('شكراً لاستخدام البرنامج!')}")
        break

    # Object detection (for all levels, but only show boxes in level 1)
    object_label = None
    object_label_arabic = None
    if model is not None:
        results = model.predict(frame, imgsz=640, conf=0.3, iou=0.4, max_det=10, verbose=False)
        
        if len(results[0].boxes) > 0:
            best_box = results[0].boxes[0]
            cls_id = int(best_box.cls[0])
            confidence = float(best_box.conf[0])
            object_label = model.names[cls_id]
            object_label_arabic = object_names_arabic.get(object_label, object_label)

            # Only show bounding boxes and labels in level 1
            if level == 1:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display Arabic object name using PIL
                label_text = f"{object_label_arabic}: {confidence:.2f}"
                frame = put_arabic_text_opencv(frame, label_text, (x1, y1 - 40), 30, (0, 255, 0))

    # Hand detection - use different settings for level 3
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    
    # Choose appropriate hand detection based on level
    if level == 3:
        hand_results = hands_level3.process(frame_rgb)
    else:
        hand_results = hands_general.process(frame_rgb)
    
    frame_rgb.flags.writeable = True

    total_fingers = 0
    hand_label = "لا يوجد"
    hand_label_arabic = {"Left": "يسار", "Right": "يمين"}

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for i, (hand_landmarks, handedness) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
            if level >= 2:  # Only draw landmarks for levels 2 and 3
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            total_fingers += count_fingers(hand_landmarks)
            # Get hand label (Left/Right) in Arabic
            hand_en = handedness.classification[0].label
            hand_label = hand_label_arabic.get(hand_en, hand_en)

    # Display level info on frame using Arabic
    level_texts = {
        1: "المستوى الأول - اكتشاف الأشياء",
        2: "المستوى الثاني - تحدي الأرقام", 
        3: "المستوى الثالث - اختبار الشهور"
    }
    
    frame = put_arabic_text_opencv(frame, level_texts[level], (20, 40), 35, (0, 255, 0))

    # Level-specific logic
    if level == 1:
        # Display detected object and hand info in Arabic on camera feed
        if object_label_arabic:
            object_text = f"الكائن المكتشف: {object_label_arabic}"
            frame = put_arabic_text_opencv(frame, object_text, (20, 90), 30, (255, 255, 0))
        
        hand_text = f"اليد: {hand_label}"
        frame = put_arabic_text_opencv(frame, hand_text, (20, 140), 30, (255, 255, 0))
        
        # Print object and hand detection results in terminal
        current_object = object_label_arabic if object_label_arabic else "لا يوجد"
        
        if current_object != last_state['object'] or hand_label != last_state['hand']:
            print(f"{display_arabic('الكائن المكتشف:')} {current_object} | {display_arabic('اليد:')} {hand_label}")
            last_state.update({'object': current_object, 'hand': hand_label})

    elif level == 2 and number_challenge_active:
        # Number Challenge Logic
        current_time = time.time()
        
        # Display target number and progress in Arabic
        target_text = f"الرقم المطلوب: {target_number}"
        frame = put_arabic_text_opencv(frame, target_text, (20, 90), 35, (0, 0, 255))
        
        digit_text = f"أظهر الرقم رقم {current_digit_position + 1}"
        frame = put_arabic_text_opencv(frame, digit_text, (20, 140), 30, (255, 0, 0))
        
        # Display current progress
        progress_text = "الإدخال: " + "".join(map(str, user_input_digits)) + "_" * (len(str(target_number)) - len(user_input_digits))
        frame = put_arabic_text_opencv(frame, progress_text, (20, 190), 30, (0, 255, 255))
        
        fingers_text = f"الأصابع: {total_fingers}"
        frame = put_arabic_text_opencv(frame, fingers_text, (20, 240), 30, (255, 255, 0))
        
        # Digit capture logic with stability check
        if total_fingers != last_finger_count:
            last_finger_count = total_fingers
            stable_finger_time = current_time
        elif current_time - stable_finger_time > 1.5:  # Stable for 1.5 seconds
            if total_fingers >= 0 and total_fingers <= 9 and current_digit_position < len(str(target_number)):
                target_digits = [int(d) for d in str(target_number)]
                expected_digit = target_digits[current_digit_position]
                
                if total_fingers == expected_digit:
                    user_input_digits.append(total_fingers)
                    current_digit_position += 1
                    stable_finger_time = current_time + 2  # Prevent immediate re-capture
                    
                    print(f"Digit {current_digit_position} captured: {total_fingers}")
                    
                    # Check if challenge is complete
                    if current_digit_position >= len(str(target_number)):
                        user_number = int("".join(map(str, user_input_digits)))
                        if user_number == target_number:
                            print(display_arabic("ممتاز! الرقم صحيح!"))
                            success_text = "ممتاز! الرقم صحيح!"
                            frame = put_arabic_text_opencv(frame, success_text, (20, 290), 35, (0, 255, 0))
                            number_challenge_complete = True
                            
                            # Auto-generate new challenge after 3 seconds
                            def new_challenge():
                                time.sleep(3)
                                start_number_challenge()
                            
                            threading.Thread(target=new_challenge, daemon=True).start()
                        else:
                            print(display_arabic("خطأ! حاول مرة أخرى"))
                            error_text = "خطأ! حاول مرة أخرى"
                            frame = put_arabic_text_opencv(frame, error_text, (20, 290), 35, (0, 0, 255))
                elif current_time - stable_finger_time > 3:  # Show wrong after 3 seconds of stable wrong input
                    wrong_text = f"خطأ! المطلوب: {expected_digit}"
                    frame = put_arabic_text_opencv(frame, wrong_text, (20, 340), 30, (0, 0, 255))
        
        # Instructions in Arabic
        instruction_text = "اضغط 'n' لرقم جديد"
        frame = put_arabic_text_opencv(frame, instruction_text, (20, frame.shape[0] - 60), 25, (255, 255, 255))
        
        hold_text = "أبق الأصابع ثابتة لـ 1.5 ثانية"
        frame = put_arabic_text_opencv(frame, hold_text, (20, frame.shape[0] - 30), 25, (255, 255, 255))

    elif level == 2 and not number_challenge_active:
        fingers_text = f"الأصابع: {total_fingers}"
        frame = put_arabic_text_opencv(frame, fingers_text, (20, 90), 35, (255, 0, 0))
        
        start_text = "اضغط 'w' لبدء تحدي الأرقام"
        frame = put_arabic_text_opencv(frame, start_text, (20, 140), 30, (255, 255, 255))

    elif level == 3 and quiz_mode:
        # Display month information in Arabic
        month_number = months.index(quiz_month) + 1
        current_time = time.time()
        elapsed_time = current_time - quiz_start_time
        
        # Display month name
        month_text = f"الشهر: {quiz_month}"
        frame = put_arabic_text_opencv(frame, month_text, (50, 90), 35, (0, 0, 0))
        
        # Different instructions based on month number
        if month_number <= 10:
            instruction_text = f"أظهر {number_to_arabic_words(month_number)} أصابع"
            frame = put_arabic_text_opencv(frame, instruction_text, (50, 140), 30, (0, 0, 0))
        else:
            if month_number == 11:
                instruction_text = "أظهر إصبعاً واحداً، ثم إصبعاً آخر (خلال 7 ثوان)"
            else:  # month_number == 12
                instruction_text = "أظهر إصبعاً واحداً، ثم إصبعين (خلال 7 ثوان)"
            frame = put_arabic_text_opencv(frame, instruction_text, (50, 140), 25, (0, 0, 0))
        
        # Show timer for months 11 and 12
        if month_number > 10:
            remaining_time = max(0, 7 - elapsed_time)
            timer_text = f"الوقت المتبقي: {remaining_time:.1f} ثانية"
            frame = put_arabic_text_opencv(frame, timer_text, (50, 190), 30, (255, 0, 0))
        
        # Show the question
        question_text = f"السؤال: ما رقم شهر {quiz_month}؟"
        frame = put_arabic_text_opencv(frame, question_text, (50, 240), 25, (0, 0, 0))
        
        if not quiz_answered:
            correct_answer = month_number
            
            if month_number <= 10:
                # Simple finger counting for months 1-10
                if total_fingers == correct_answer and total_fingers != 0:
                    print(display_arabic("إجابة صحيحة! أحسنت!"))
                    correct_text = "إجابة صحيحة! أحسنت!"
                    frame = put_arabic_text_opencv(frame, correct_text, (50, 290), 35, (0, 255, 0))
                    quiz_answered = True
                    # Auto-generate new question
                    threading.Thread(target=new_quiz, daemon=True).start()
                elif total_fingers != 0 and elapsed_time > 3:  # Give some time before showing wrong
                    print(f"{display_arabic('إجابة خاطئة! الإجابة الصحيحة هي:')} {correct_answer}")
                    wrong_text = f"إجابة خاطئة! الإجابة الصحيحة: {correct_answer}"
                    frame = put_arabic_text_opencv(frame, wrong_text, (50, 290), 30, (0, 0, 255))
                    quiz_answered = True
                    # Auto-generate new question
                    threading.Thread(target=new_quiz, daemon=True).start()
            
            else:
                # Two-gesture system for months 11 and 12
                if elapsed_time <= 7:
                    if total_fingers > 0:
                        if first_gesture_time == 0:
                            # Record first gesture
                            first_gesture_time = current_time
                            first_gesture_count = total_fingers
                            print(f"First gesture recorded: {first_gesture_count} fingers")
                        elif (current_time - first_gesture_time) > 1:  # At least 1 second between gestures
                            # Check second gesture
                            second_gesture_count = total_fingers
                            combined_number = first_gesture_count * 10 + second_gesture_count
                            
                            if combined_number == correct_answer:
                                print(display_arabic("إجابة صحيحة! أحسنت!"))
                                correct_text = "إجابة صحيحة! أحسنت!"
                                frame = put_arabic_text_opencv(frame, correct_text, (50, 290), 35, (0, 255, 0))
                                quiz_answered = True
                                # Auto-generate new question
                                threading.Thread(target=new_quiz, daemon=True).start()
                            else:
                                print(f"{display_arabic('إجابة خاطئة! الإجابة الصحيحة هي:')} {correct_answer}")
                                wrong_text = f"إجابة خاطئة! الإجابة الصحيحة: {correct_answer}"
                                frame = put_arabic_text_opencv(frame, wrong_text, (50, 290), 30, (0, 0, 255))
                                quiz_answered = True
                                # Auto-generate new question
                                threading.Thread(target=new_quiz, daemon=True).start()
                else:
                    # Time's up
                    print(f"{display_arabic('انتهى الوقت! الإجابة الصحيحة هي:')} {correct_answer}")
                    timeout_text = f"انتهى الوقت! الإجابة الصحيحة: {correct_answer}"
                    frame = put_arabic_text_opencv(frame, timeout_text, (50, 290), 30, (0, 0, 255))
                    quiz_answered = True
                    # Auto-generate new question
                    threading.Thread(target=new_quiz, daemon=True).start()

    # *** DISPLAY THE CAMERA FEED - THIS IS THE KEY LINE ***
    cv2.imshow("Object Detection Camera", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(display_arabic("تم إغلاق البرنامج بنجاح"))