from data_loader.base_loader import BaseLoader
import logging
no_cot_examples = [
"""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: The answer is 11.
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?""",
"""Q: Jane has 8 marbles. She gives 3 to her friend and then buys 5 more. How many marbles does she have now?
A: The answer is 10.
Q: Alex had 15 pencils. He lost 7 and then found 4 more. How many pencils does he have now?""",
"""Q: Emily had 30 stickers. She gave away 10 and then received 8 as a gift. How many stickers does she have now?
A: The answer is 28.
Q: Michael had 50 toy cars. He sold 15 and then bought 20 more. How many toy cars does he have now?""",
"""Q: Sarah had 12 cookies. She ate 5 and then baked 8 more. How many cookies does she have now?
A: The answer is 15.
Q: David had 25 balloons. He popped 9 and then inflated 6 more. How many balloons does he have now?""",
"""Q: Lily had 40 candies. She shared 15 with her friends and then bought 12 more. How many candies does she have now?
A: The answer is 37.
Q: Mark had 18 baseball cards. He traded 8 and then received 10 new ones. How many baseball cards does he have now?""",
"""Q: Anna had 60 books. She donated 25 and then bought 30 more. How many books does she have now?
A: The answer is 65.
Q: Peter had 100 coins. He spent 40 and then found 25 more. How many coins does he have now?""",
"""Q: Lucy had 8 pencils. She lost 3 of them and then found 5 more. How many pencils does she have now?
A: The answer is 10.
Q: Jason had 15 marbles. He gave away 7 and then bought 4 more bags of marbles. Each bag contained 6 marbles. How many marbles does he have now?""",
"""Q: Emily had 40 candies. She ate 15 and then received 10 more as a gift. How many candies does she have now?
A: The answer is 35.
Q: Alex had $50. He spent $20 on a book and then earned $30 from his part-time job. How much money does he have now?""",
"""Q: Sarah had 25 stickers. She gave away 10 and then found 15 more on the playground. How many stickers does she have now?
A: The answer is 30.
Q: David had 18 toy cars. He sold 5 at a garage sale and then received 7 more as a birthday gift. How many toy cars does he have now?""",
"""Q: Tina had 60 puzzle pieces. She used 25 to complete a puzzle and then bought 30 more. How many puzzle pieces does she have now?
A: The answer is 65.
Q: Michael had 12 baseball cards. He traded 4 with his friend and then received 8 more as a reward. How many baseball cards does he have now?""",
"""Q: Laura had 30 balloons. She gave away 12 and then bought 20 more for the party. How many balloons does she have now?
A: The answer is 38.
Q: Peter had 100 LEGO bricks. He used 50 to build a castle and then received 40 more as a gift. How many LEGO bricks does he have now?"""
]
cot_examples = [
"""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?""",
"""Q: Jane has 8 marbles. She gives 3 to her friend and then buys 5 more. How many marbles does she have now?
A: Jane started with 8 marbles. Giving away 3 leaves her with 5 marbles. Buying 5 more adds up to 10 marbles. 5 + 5 = 10. The answer is 10.
Q: Alex had 15 pencils. He lost 7 and then found 4 more. How many pencils does he have now?""",
"""Q: Emily had 30 stickers. She gave away 10 and then received 8 as a gift. How many stickers does she have now?
A: Emily started with 30 stickers. Giving away 10 leaves her with 20 stickers. Receiving 8 more adds up to 28 stickers. 20 + 8 = 28. The answer is 28.
Q: Michael had 50 toy cars. He sold 15 and then bought 20 more. How many toy cars does he have now?""",
"""Q: Sarah had 12 cookies. She ate 5 and then baked 8 more. How many cookies does she have now?
A: Sarah started with 12 cookies. Eating 5 leaves her with 7 cookies. Baking 8 more adds up to 15 cookies. 7 + 8 = 15. The answer is 15.
Q: David had 25 balloons. He popped 9 and then inflated 6 more. How many balloons does he have now?""",
"""Q: Lily had 40 candies. She shared 15 with her friends and then bought 12 more. How many candies does she have now?
A: Lily started with 40 candies. Sharing 15 leaves her with 25 candies. Buying 12 more adds up to 37 candies. 25 + 12 = 37. The answer is 37.
Q: Mark had 18 baseball cards. He traded 8 and then received 10 new ones. How many baseball cards does he have now?""",
"""Q: Anna had 60 books. She donated 25 and then bought 30 more. How many books does she have now?
A: Anna started with 60 books. Donating 25 leaves her with 35 books. Buying 30 more adds up to 65 books. 35 + 30 = 65. The answer is 65.
Q: Peter had 100 coins. He spent 40 and then found 25 more. How many coins does he have now?""",
"""Q: Lucy had 8 pencils. She lost 3 of them and then found 5 more. How many pencils does she have now?
A: Lucy began with 8 pencils. Losing 3 leaves her with 5. Finding 5 more pencils, she now has 10. The answer is 10.
Q: Jason had 15 marbles. He gave away 7 and then bought 4 more bags of marbles. Each bag contained 6 marbles. How many marbles does he have now?""",
"""Q: Emily had 40 candies. She ate 15 and then received 10 more as a gift. How many candies does she have now?
A: Emily started with 40 candies. Eating 15 leaves her with 25. Receiving 10 more candies, she now has 35. The answer is 35.
Q: Alex had $50. He spent $20 on a book and then earned $30 from his part-time job. How much money does he have now?""",
"""Q: Sarah had 25 stickers. She gave away 10 and then found 15 more on the playground. How many stickers does she have now?
A: Sarah initially had 25 stickers. Giving away 10 leaves her with 15. Finding 15 more stickers, she now has 30. The answer is 30.
Q: David had 18 toy cars. He sold 5 at a garage sale and then received 7 more as a birthday gift. How many toy cars does he have now?""",
"""Q: Tina had 60 puzzle pieces. She used 25 to complete a puzzle and then bought 30 more. How many puzzle pieces does she have now?
A: Tina initially had 60 puzzle pieces. Using 25 leaves her with 35. Buying 30 more puzzle pieces, she now has 65. The answer is 65.
Q: Michael had 12 baseball cards. He traded 4 with his friend and then received 8 more as a reward. How many baseball cards does he have now?""",
"""Q: Laura had 30 balloons. She gave away 12 and then bought 20 more for the party. How many balloons does she have now?
A: Laura initially had 30 balloons. Giving away 12 leaves her with 18. Buying 20 more balloons, she now has 38. The answer is 38.
Q: Peter had 100 LEGO bricks. He used 50 to build a castle and then received 40 more as a gift. How many LEGO bricks does he have now?"""
]
cot_shut_examples = [
"""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?""",
"""Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: Laura had 30 balloons. She gave away 12 and then bought 20 more for the party. How many balloons does she have now?
A: Laura initially had 30 balloons. Giving away 12 leaves her with 18. Buying 20 more balloons, she now has 38. The answer is 38.
Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?""",
"""Q: Jane has 8 marbles. She gives 3 to her friend and then buys 5 more. How many marbles does she have now?
A: Jane started with 8 marbles. Giving away 3 leaves her with 5 marbles. Buying 5 more adds up to 10 marbles. 5 + 5 = 10. The answer is 10.
Q: Tina had 60 puzzle pieces. She used 25 to complete a puzzle and then bought 30 more. How many puzzle pieces does she have now?
A: Tina initially had 60 puzzle pieces. Using 25 leaves her with 35. Buying 30 more puzzle pieces, she now has 65. The answer is 65.
Q: Emily had 40 candies. She ate 15 and then received 10 more as a gift. How many candies does she have now?
A: Emily started with 40 candies. Eating 15 leaves her with 25. Receiving 10 more candies, she now has 35. The answer is 35.
Q: Alex had 15 pencils. He lost 7 and then found 4 more. How many pencils does he have now?""",
"""Q: Emily had 30 stickers. She gave away 10 and then received 8 as a gift. How many stickers does she have now?
A: Emily started with 30 stickers. Giving away 10 leaves her with 20 stickers. Receiving 8 more adds up to 28 stickers. 20 + 8 = 28. The answer is 28.
Q: Michael had 50 toy cars. He sold 15 and then bought 20 more. How many toy cars does he have now?""",
"""Q: Tina had 60 puzzle pieces. She used 25 to complete a puzzle and then bought 30 more. How many puzzle pieces does she have now?
A: Tina initially had 60 puzzle pieces. Using 25 leaves her with 35. Buying 30 more puzzle pieces, she now has 65. The answer is 65.
Q: Laura had 30 balloons. She gave away 12 and then bought 20 more for the party. How many balloons does she have now?
A: Laura initially had 30 balloons. Giving away 12 leaves her with 18. Buying 20 more balloons, she now has 38. The answer is 38.
Q: Peter had 100 LEGO bricks. He used 50 to build a castle and then received 40 more as a gift. How many LEGO bricks does he have now?"""
]

class CotLoader(BaseLoader):
    def __init__(self):
        self.name = "Cot_examples"

    def load_data(self, data_len = 200):
        logging.info(f"Loading data {self.name} ...") 
        
        # Cot_examples = no_cot_examples+cot_examples + cot_shut_examples
        Cot_examples = [element for trio in zip(no_cot_examples, cot_examples, cot_examples) for element in trio]
        return Cot_examples[:data_len]
