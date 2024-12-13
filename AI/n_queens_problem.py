n = 8 # Number of queens and rows/cols.
res = []
board = [["."] * n for _ in range(n)]

col = set()
negDiag = set()
posDiag = set()

def backtrack(r):
    # If the decision tree reached this leaf, it means one valid solution was found.
    if r == n:
        copy = ["".join(row) for row in board]
        res.append(copy) 
        return
    
    for c in range(n):
        """
        posDiag -> r+c is always the same in this type of diagonal.
        negDiag -> r-c is always the same in this type of diagonal.
        """
        if c in col or (r+c) in posDiag or (r-c) in negDiag:
            continue
        
        col.add(c)
        negDiag.add(r-c)
        posDiag.add(r+c)
        board[r][c] = "Q"
        
        backtrack(r+1)
        
        col.remove(c)
        negDiag.remove(r-c)
        posDiag.remove(r+c)
        board[r][c] = "."
        
backtrack(0)
print(res)