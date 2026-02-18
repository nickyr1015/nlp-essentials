# src/homework/text_processing.py

import re
import calendar
import pprint

def chronicles_of_narnia(file_path="dat/chronicles_of_narnia.txt"):

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    
    books = {}
    current_book = None
    current_chapter = None
    current_tokens = 0
    
    # Pattern to match chapter lines
    chapter_pattern = re.compile(r"^\s*chapter\s+(.+)\s*$", re.IGNORECASE)
    
    # Pattern to find years in parentheses: (year)
    year_pattern = re.compile(r"\((\s*(1[6-9]\d{2}|20\d{2})\s*)\)")
    
    def roman_to_int(roman):
        roman = roman.strip().upper()
        if roman.isdigit():
            return int(roman)
        
        roman_numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        if not roman:
            return None
        
        for char in roman:
            if char not in roman_numerals:
                return None
        
        total = 0
        prev_value = 0
        for char in reversed(roman):
            value = roman_numerals[char]
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        return total
    
    def is_book_heading(line):
        if not line or chapter_pattern.match(line):
            return False
        
        # Book titles will always have a year in parenthese
        return year_pattern.search(line) is not None
    
    def get_next_nonempty_line(start_idx):
        """Find the next non-empty line after start_idx."""
        for i in range(start_idx + 1, len(lines)):
            if lines[i]:
                return i
        return None
    
    # Process each line
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this is a book title heading!
        if is_book_heading(line):
            next_idx = get_next_nonempty_line(i)
            if next_idx is not None and chapter_pattern.match(lines[next_idx]):
                # Save previous chapter if exists
                if current_book and current_chapter:
                    current_chapter["token_count"] = current_tokens
                    books[current_book]["chapters"].append(current_chapter)
                
                # Extract book title and year from the line
                year_match = year_pattern.search(line)
                book_year = None
                book_title = line.strip()
                
                if year_match:
                    # Extract year from the match
                    year_str = year_match.group(2)
                    book_year = int(year_str)
                    # Remove the (year) part to get just the title
                    book_title = year_pattern.sub("", line).strip()
                
                # Use the cleaned title as the dictionary key
                current_book = book_title
                if current_book not in books:
                    books[current_book] = {"title": book_title, "year": book_year, "chapters": []}
                
                current_chapter = None
                current_tokens = 0
                i = i + 1
                continue
        
        # Check if this is a chapter line
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            # Save previous chapter
            if current_book and current_chapter:
                current_chapter["token_count"] = current_tokens
                books[current_book]["chapters"].append(current_chapter)
            
            # Parse chapter number
            chapter_text = chapter_match.group(1).strip()
            parts = chapter_text.split()
            
            # Get chapter number
            chapter_num = None
            if parts:
                # Clean the first part (remove punctuation)
                first_part = parts[0].strip(".:;,-_()[]{}")
                chapter_num = roman_to_int(first_part)
            
            # If we couldn't parse the number, use sequential numbering
            if chapter_num is None:
                if current_book and books[current_book]["chapters"]:
                    last_num = books[current_book]["chapters"][-1]["number"]
                    chapter_num = last_num + 1
                else:
                    chapter_num = 1
            
            # Get chapter title from the next non-empty line
            chapter_title = ""
            next_line_idx = get_next_nonempty_line(i)
            if next_line_idx is not None:
                next_line = lines[next_line_idx]
                # If the next line is not another chapter line, it's the title
                if not chapter_pattern.match(next_line):
                    chapter_title = next_line.strip()
            
            # If no book yet, create a default one
            if current_book is None:
                current_book = "UNKNOWN_BOOK"
                books[current_book] = {"title": current_book, "year": None, "chapters": []}
            
            # Start new chapter
            current_chapter = {"number": chapter_num, "title": chapter_title, "token_count": 0}
            current_tokens = 0
            
            # Skip the chapter line
            i += 1
            # If we found a title, also skip the title line
            if chapter_title and next_line_idx is not None:
                i = next_line_idx + 1
            continue
        
        # Regular content line - count tokens
        if current_book and current_chapter:
            tokens = line.split()
            current_tokens += len(tokens)
        
        i += 1
    
    # Save the last chapter
    if current_book and current_chapter:
        current_chapter["token_count"] = current_tokens
        books[current_book]["chapters"].append(current_chapter)
    
    # Sort chapters by number within each book
    for book_title in books:
        books[book_title]["chapters"].sort(key=lambda ch: ch["number"])
    
    return books


def regular_expressions(text):


    # -------------------
    # Email
    # username@hostname.domain
    # username/hostname: letters, numbers, . _ -
    # must start/end with letter/number
    # domain: com|org|edu|gov
    # -------------------
    email_pattern = re.compile(
        r'^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?@'
        r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?'
        r'\.(com|org|edu|gov)$'
    )
    if email_pattern.fullmatch(text):
        return "email"

    # -------------------
    # Date
    # YYYY/MM/DD or YY/MM/DD
    # YYYY-MM-DD or YY-MM-DD
    # 4-digit year: 1951-2050
    # 2-digit year: corresponds to 1951-2050 (51-99 => 1951-1999, 00-50 => 2000-2050)
    # month: 1-12 (optional leading 0)
    # day: 1-31 (optional leading 0) but must be valid for month/year
    # -------------------
    date_pattern = re.compile(
        r'^(?P<year4>19[5-9]\d|20[0-4]\d|2050|(?P<year2>\d{2}))'
        r'(?P<sep>[/-])'
        r'(?P<month>0?[1-9]|1[0-2])'
        r'(?P=sep)'
        r'(?P<day>0?[1-9]|[12]\d|3[01])$'
    )
    m = date_pattern.fullmatch(text)
    if m:
        if m.group("year4") and len(m.group("year4")) == 4:
            year = int(m.group("year4"))
        else:
            yy = int(m.group("year2"))
            year = 1900 + yy if yy >= 51 else 2000 + yy

        month = int(m.group("month"))
        day = int(m.group("day"))

        if 1951 <= year <= 2050:
            try:
                if 1 <= day <= calendar.monthrange(year, month)[1]:
                    return "date"
            except:
                pass

    # -------------------
    # URL
    # protocol://address
    # protocol: http|https
    # address: letters, hyphen, dots ONLY
    # must start with letter/number
    # must include at least one dot
    # -------------------
    url_pattern = re.compile(
        r'^(http|https)://'
        r'[A-Za-z0-9][A-Za-z.-]*'
        r'\.[A-Za-z.-]+$'
    )
    if url_pattern.fullmatch(text):
        return "url"

    # -------------------
    # Cite
    # Single: Lastname, YYYY
    # Two: Lastname1 and Lastname2, YYYY
    # Multi: Lastname1 et al., YYYY
    # Lastnames capitalized and can have multiple parts (e.g., "Van Helsing")
    # Year 1900-2024
    # -------------------
    name = r'[A-Z][a-zA-Z]*'                 # one capitalized word
    lastname = rf'{name}(?:\s+{name})*'      # allow multiple words, each capitalized

    cite_pattern = re.compile(
        rf'^(?:'
        rf'(?P<a1>{lastname})'                                   # single author
        rf'|'
        rf'(?P<a2_1>{lastname})\s+and\s+(?P<a2_2>{lastname})'     # two authors
        rf'|'
        rf'(?P<a3>{lastname})\s+et\s+al\.'                        # multiple authors
        rf'),\s+(?P<year>19\d{{2}}|20(?:0\d|1\d|2[0-4]))$'
    )
    cm = cite_pattern.fullmatch(text)
    if cm:
        year = int(cm.group("year"))
        if 1900 <= year <= 2024:
            return "cite"

    return None



if __name__ == "__main__":
   
    data = chronicles_of_narnia("dat/chronicles_of_narnia.txt")
    pprint.pprint(data, width=100, sort_dicts=False)

    tests = [
    # ---------- email (valid) ----------
    "a@b.com",
    "john.doe_2@my-host.edu",
    "x_y-z.9@abc_def.gov",

    # ---------- email (invalid) ----------
    ".abc@host.com",          # starts with .
    "abc.@host.com",          # ends with .
    "abc@-host.com",          # hostname starts with -
    "abc@host.xyz",           # bad domain
    "abc@host..com",          # double dot (should fail with our stricter assumptions? depends on regex)

    # ---------- date (valid) ----------
    "1951/1/1",
    "2050-12-31",
    "99/02/28",               # -> 1999-02-28
    "00-2-29",                # -> 2000-02-29 (leap year)

    # ---------- date (invalid) ----------
    "50/12/31",               # -> 2050 ok actually (this one is valid under your mapping)
    "51/13/01",               # month 13
    "51/00/10",               # month 0
    "2023/02/29",             # not leap year
    "2051-01-01",             # year out of range
    "1950-12-31",             # year out of range
    "1999-04-31",             # April has 30 days

    # ---------- url (valid) ----------
    "http://a.b",
    "https://narnia.com",
    "https://my-site.example",  # NOTE: this will FAIL because we require only letters/hyphen/dots; "example" ok, but TLD doesn't need to be real
    "http://abc-def.ghi.jkl",

    # ---------- url (invalid) ----------
    "ftp://a.b",              # bad protocol
    "http://-abc.com",        # starts with -
    "http://abc",             # no dot
    "https://ab_c.com",       # underscore not allowed in URL address
    "https://a..b",           # double dot (your regex may allow; depends how strict you want)

    # ---------- cite (valid) ----------
    "Smith, 2023",
    "Smith and Jones, 2023",
    "Smith et al., 2023",
    "Van Helsing, 1900",
    "De La Cruz and Smith, 2024",

    # ---------- cite (invalid) ----------
    "smith, 2023",            # not capitalized
    "Smith and jones, 2023",  # second not capitalized
    "Smith et al, 2023",      # missing period after al.
    "Smith and Jones et al., 2023",  # not allowed format
    "Smith, 1899",            # year too early
    "Smith, 2025",            # year too late
]

# for t in tests:
#     print(f"{t!r:35} -> {regular_expressions(t)}")

