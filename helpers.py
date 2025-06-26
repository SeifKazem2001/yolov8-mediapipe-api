import time
import random
import threading
from collections import deque
#from utils import number_to_arabic_words, display_arabic, speak_arabic
# Enhanced finger stabilization class with better detection
class FingerStabilizer:
    def __init__(self, window_size=15, stability_threshold=0.85, min_stable_time=2.0):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.min_stable_time = min_stable_time
        self.finger_history = deque(maxlen=window_size)
        self.last_stable_count = -1
        self.stable_start_time = 0
        self.is_stable = False
        self.confidence_scores = deque(maxlen=window_size)
        
    def update(self, finger_count, confidence=1.0):
        """Update with new finger count and confidence score"""
        current_time = time.time()
        self.finger_history.append(finger_count)
        self.confidence_scores.append(confidence)
        
        if len(self.finger_history) < self.window_size:
            return None
            
        # Calculate weighted frequency based on confidence
        count_weights = {}
        for i, count in enumerate(self.finger_history):
            weight = self.confidence_scores[i]
            if count not in count_weights:
                count_weights[count] = 0
            count_weights[count] += weight
            
        # Find most frequent count with highest weighted score
        total_weight = sum(self.confidence_scores)
        most_frequent_count = max(count_weights, key=count_weights.get)
        frequency_ratio = count_weights[most_frequent_count] / total_weight
        
        # Enhanced stability check
        if frequency_ratio >= self.stability_threshold:
            if most_frequent_count != self.last_stable_count:
                self.last_stable_count = most_frequent_count
                self.stable_start_time = current_time
                self.is_stable = False
            elif current_time - self.stable_start_time >= self.min_stable_time:
                if not self.is_stable:
                    self.is_stable = True
                    return most_frequent_count
        else:
            self.is_stable = False
            
        return None
    
    def get_current_stable_count(self):
        """Get the current stable count without triggering new detection"""
        return self.last_stable_count if self.is_stable else None
    
    def get_stability_percentage(self):
        """Get current stability percentage for display"""
        if len(self.finger_history) == 0:
            return 0
        
        count_frequency = {}
        for count in self.finger_history:
            count_frequency[count] = count_frequency.get(count, 0) + 1
            
        if not count_frequency:
            return 0
            
        most_frequent = max(count_frequency.values())
        return (most_frequent / len(self.finger_history)) * 100
    
    def reset(self):
        """Reset the stabilizer"""
        self.finger_history.clear()
        self.confidence_scores.clear()
        self.last_stable_count = -1
        self.stable_start_time = 0
        self.is_stable = False

# Enhanced Progressive Number Challenge with proper level progression
class ProgressiveNumberChallenge:
    def __init__(self):
        self.current_level = 1  # Start with 1-digit numbers
        self.max_level = 5      # Up to 5-digit numbers
        self.challenges_completed = 0
        self.challenges_needed_to_advance = 3  # Need 3 correct answers to advance
        self.target_number = 0
        self.user_input_digits = []
        self.current_digit_position = 0
        self.challenge_active = False
        self.challenge_complete = False
        self.stabilizer = FingerStabilizer(window_size=12, stability_threshold=0.8, min_stable_time=1.8)
        self.last_detection_time = 0
        self.detection_cooldown = 1.0  # Cooldown between detections
        
    def get_number_range(self):
        """Get number range based on current level"""
        ranges = {
            1: (1, 9),          # 1-digit: 1-9
            2: (10, 99),        # 2-digit: 10-99
            3: (100, 999),      # 3-digit: 100-999
            4: (1000, 9999),    # 4-digit: 1000-9999
            5: (10000, 99999)   # 5-digit: 10000-99999
        }
        return ranges.get(self.current_level, (1, 9))
    
    def start_new_challenge(self):
        """Start a new number challenge"""
        min_num, max_num = self.get_number_range()
        self.target_number = random.randint(min_num, max_num)
        self.user_input_digits = []
        self.current_digit_position = 0
        self.challenge_active = True
        self.challenge_complete = False
        self.stabilizer.reset()
        self.last_detection_time = 0
        
        # Convert number to Arabic and speak it
        arabic_text = number_to_arabic_words(self.target_number)
        level_text = f"المستوى {self.current_level} - رقم من {len(str(self.target_number))} أرقام"
        progress_text = f"التقدم: {self.challenges_completed}/{self.challenges_needed_to_advance}"
        
        print(f"\n{display_arabic(level_text)}")
        print(f"{display_arabic(progress_text)}")
        print(f"{display_arabic('رقم جديد:')} {self.target_number}")
        print(f"{display_arabic('بالعربية:')} {arabic_text}")
        print(f"{display_arabic('أظهر كل رقم بأصابعك بالترتيب من اليسار إلى اليمين')}")
        
        # Speak the number in a separate thread
        def speak_thread():
            speak_arabic(arabic_text)
        
        threading.Thread(target=speak_thread, daemon=True).start()
    
    def process_finger_input(self, finger_count, confidence=1.0):
        """Enhanced finger input processing with better detection"""
        if not self.challenge_active or self.challenge_complete:
            return None
            
        current_time = time.time()
        
        # Apply cooldown to prevent rapid multiple detections
        if current_time - self.last_detection_time < self.detection_cooldown:
            return None
            
        # Use enhanced stabilizer
        stable_count = self.stabilizer.update(finger_count, confidence)
        
        if stable_count is not None:
            target_digits = [int(d) for d in str(self.target_number)]
            
            if self.current_digit_position < len(target_digits):
                expected_digit = target_digits[self.current_digit_position]
                
                if stable_count == expected_digit:
                    # Correct digit!
                    self.user_input_digits.append(stable_count)
                    self.current_digit_position += 1
                    self.last_detection_time = current_time
                    
                    print(f"✓ الرقم {self.current_digit_position} صحيح: {stable_count}")
                    
                    # Check if challenge is complete
                    if self.current_digit_position >= len(target_digits):
                        self.challenge_complete = True
                        self.challenges_completed += 1
                        
                        print(display_arabic("🎉 ممتاز! الرقم كامل صحيح!"))
                        
                        # Check for level advancement
                        if self.challenges_completed >= self.challenges_needed_to_advance:
                            if self.current_level < self.max_level:
                                self.current_level += 1
                                self.challenges_completed = 0
                                print(f"🎊 {display_arabic('تقدمت للمستوى')} {self.current_level}!")
                                print(f"{display_arabic('الآن ستتعامل مع أرقام من')} {len(str(self.get_number_range()[1]))} {display_arabic('خانات')}")
                            else:
                                print(display_arabic("🏆 تهانينا! أكملت جميع المستويات!"))
                        
                        # Auto-generate new challenge after 3 seconds
                        def new_challenge():
                            time.sleep(3)
                            if self.current_level <= self.max_level:
                                self.start_new_challenge()
                        
                        threading.Thread(target=new_challenge, daemon=True).start()
                        return "correct"
                        
                elif stable_count >= 0 and stable_count <= 9:
                    # Wrong digit - give specific feedback
                    expected_digit = target_digits[self.current_digit_position]
                    print(f"❌ {display_arabic('خطأ!')} {display_arabic('المطلوب:')} {expected_digit}, {display_arabic('أظهرت:')} {stable_count}")
                    
                    # Reset to beginning of current number
                    self.user_input_digits = []
                    self.current_digit_position = 0
                    self.stabilizer.reset()
                    self.last_detection_time = current_time
                    
                    return "wrong"
        
        return None
    
    def get_progress_display(self):
        """Get visual progress display"""
        target_str = str(self.target_number)
        progress_display = ""
        
        for i, digit in enumerate(target_str):
            if i < len(self.user_input_digits):
                progress_display += f"✓{self.user_input_digits[i]} "
            elif i == self.current_digit_position:
                progress_display += f"_{digit}_ "
            else:
                progress_display += f"?{digit}? "
                
        return progress_display.strip()

# Enhanced Month Quiz with better stabilization
class MonthQuizChallenge:
    def __init__(self):
        self.quiz_active = False
        self.quiz_answered = False
        self.quiz_month = ""
        self.quiz_start_time = 0
        self.first_gesture_time = 0
        self.first_gesture_count = 0
        self.stabilizer = FingerStabilizer(window_size=15, stability_threshold=0.8, min_stable_time=2.2)
        self.last_detection_time = 0
        self.detection_cooldown = 1.5
        self.months = [
            "محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جمادى الأولى", "جمادى الآخرة",
            "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة"
        ]
        self.correct_answers = 0
        self.total_questions = 0
        
    def start_new_quiz(self):
        """Start a new month quiz"""
        time.sleep(3)
        self.quiz_active = True
        self.quiz_answered = False
        self.quiz_month = random.choice(self.months)
        self.quiz_start_time = time.time()
        self.first_gesture_time = 0
        self.first_gesture_count = 0
        self.stabilizer.reset()
        self.last_detection_time = 0
        self.total_questions += 1

        # Process month name for display
        shaped_month = display_arabic(self.quiz_month)
        month_number = self.months.index(self.quiz_month) + 1
        
        print(f"\n📅 {display_arabic('سؤال رقم')} {self.total_questions}")
        print(f"🎯 {display_arabic('النتيجة:')} {self.correct_answers}/{self.total_questions-1}")
        print(f"📝 {display_arabic('الشهر:')} {shaped_month}")
        print(f"🔢 {display_arabic('أظهر رقم الشهر بأصابعك:')} {month_number}")
        
        if month_number > 10:
            print(f"⚠️ {display_arabic('للأشهر 11 و 12: أظهر الرقم الأول ثم الرقم الثاني خلال 15 ثانية')}")
            print(f"   {display_arabic('مثال: للشهر 11 أظهر 1 ثم 1')}")
            print(f"   {display_arabic('مثال: للشهر 12 أظهر 1 ثم 2')}")
    
    def process_finger_input(self, finger_count, confidence=1.0):
        """Enhanced finger input processing for month quiz"""
        if not self.quiz_active or self.quiz_answered:
            return None
            
        current_time = time.time()
        elapsed_time = current_time - self.quiz_start_time
        month_number = self.months.index(self.quiz_month) + 1
        
        # Apply cooldown
        if current_time - self.last_detection_time < self.detection_cooldown:
            return None
        
        if month_number <= 10:
            # Simple single gesture for months 1-10
            stable_count = self.stabilizer.update(finger_count, confidence)
            
            if stable_count is not None and stable_count > 0:
                self.last_detection_time = current_time
                
                if stable_count == month_number:
                    self.correct_answers += 1
                    print(f"✅ {display_arabic('إجابة صحيحة! أحسنت!')}")
                    print(f"🎯 {display_arabic('النتيجة الجديدة:')} {self.correct_answers}/{self.total_questions}")
                    self.quiz_answered = True
                    threading.Thread(target=self.start_new_quiz, daemon=True).start()
                    return "correct"
                else:
                    print(f"❌ {display_arabic('إجابة خاطئة!')}")
                    print(f"📝 {display_arabic('الإجابة الصحيحة:')} {month_number}")
                    print(f"🎯 {display_arabic('النتيجة:')} {self.correct_answers}/{self.total_questions}")
                    self.quiz_answered = True
                    threading.Thread(target=self.start_new_quiz, daemon=True).start()
                    return "wrong"
        else:
            # Enhanced two-gesture system for months 11 and 12
            if elapsed_time <= 15:  # Extended time limit
                stable_count = self.stabilizer.update(finger_count, confidence)
                
                if stable_count is not None and stable_count > 0:
                    if self.first_gesture_time == 0:
                        # Record first gesture
                        self.first_gesture_time = current_time
                        self.first_gesture_count = stable_count
                        self.stabilizer.reset()
                        self.last_detection_time = current_time
                        
                        print(f"1️⃣ {display_arabic('الرقم الأول:')} {stable_count}")
                        print(f"⏳ {display_arabic('الآن أظهر الرقم الثاني...')}")
                        return "first_gesture"
                        
                    elif (current_time - self.first_gesture_time) > 3:  # At least 3 seconds between gestures
                        # Check second gesture
                        second_gesture_count = stable_count
                        combined_number = self.first_gesture_count * 10 + second_gesture_count
                        self.last_detection_time = current_time
                        
                        print(f"2️⃣ {display_arabic('الرقم الثاني:')} {second_gesture_count}")
                        print(f"🔢 {display_arabic('الرقم الكامل:')} {combined_number}")
                        
                        if combined_number == month_number:
                            self.correct_answers += 1
                            print(f"✅ {display_arabic('إجابة صحيحة! ممتاز!')}")
                            print(f"🎯 {display_arabic('النتيجة الجديدة:')} {self.correct_answers}/{self.total_questions}")
                            self.quiz_answered = True
                            threading.Thread(target=self.start_new_quiz, daemon=True).start()
                            return "correct"
                        else:
                            print(f"❌ {display_arabic('إجابة خاطئة!')}")
                            print(f"📝 {display_arabic('الإجابة الصحيحة:')} {month_number}")
                            print(f"🎯 {display_arabic('النتيجة:')} {self.correct_answers}/{self.total_questions}")
                            self.quiz_answered = True
                            threading.Thread(target=self.start_new_quiz, daemon=True).start()
                            return "wrong"
            else:
                # Time's up
                print(f"⏰ {display_arabic('انتهى الوقت!')}")
                print(f"📝 {display_arabic('الإجابة الصحيحة:')} {month_number}")
                print(f"🎯 {display_arabic('النتيجة:')} {self.correct_answers}/{self.total_questions}")
                self.quiz_answered = True
                threading.Thread(target=self.start_new_quiz, daemon=True).start()
                return "timeout"
        
        return None
    
    def get_time_remaining(self):
        """Get remaining time for current quiz"""
        if not self.quiz_active:
            return 0
        elapsed = time.time() - self.quiz_start_time
        month_number = self.months.index(self.quiz_month) + 1
        time_limit = 15 if month_number > 10 else 10
        return max(0, time_limit - elapsed)

# Enhanced finger counting with confidence scoring
def count_fingers_enhanced(hand_landmarks):
    """Enhanced finger counting with confidence scoring"""
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [2, 6, 10, 14, 18]  # Joint below tips
    finger_mcp = [1, 5, 9, 13, 17]    # Knuckles
    
    count = 0
    confidence_scores = []
    
    # Thumb (special case - check x-coordinate)
    thumb_tip = hand_landmarks.landmark[finger_tips[0]]
    thumb_pip = hand_landmarks.landmark[finger_pips[0]]
    thumb_mcp = hand_landmarks.landmark[finger_mcp[0]]
    
    # Enhanced thumb detection
    thumb_extended = False
    thumb_distance = abs(thumb_tip.x - thumb_mcp.x)
    thumb_pip_distance = abs(thumb_pip.x - thumb_mcp.x)
    
    if thumb_distance > thumb_pip_distance * 1.2:  # More strict threshold
        thumb_extended = True
        confidence_scores.append(min(1.0, thumb_distance / thumb_pip_distance))
    
    if thumb_extended:
        count += 1
    
    # Other fingers with enhanced detection
    for i in range(1, 5):
        tip = hand_landmarks.landmark[finger_tips[i]]
        pip = hand_landmarks.landmark[finger_pips[i]]
        mcp = hand_landmarks.landmark[finger_mcp[i]]
        
        # Enhanced finger detection with angle consideration
        tip_pip_dist = abs(tip.y - pip.y)
        pip_mcp_dist = abs(pip.y - mcp.y)
        
        # Check if finger is extended with confidence
        if tip.y < pip.y and pip.y < mcp.y and tip_pip_dist > 0.02:
            count += 1
            # Calculate confidence based on extension ratio
            extension_ratio = tip_pip_dist / max(pip_mcp_dist, 0.01)
            confidence_scores.append(min(1.0, extension_ratio))
    
    # Calculate overall confidence
    overall_confidence = sum(confidence_scores) / max(len(confidence_scores), 1) if confidence_scores else 0.5
    
    return count, overall_confidence