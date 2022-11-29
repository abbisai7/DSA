def isOperator(curChar):
    operators =  ["+","-","*","/","^","%"]
    if curChar in operators:
        return True
    return False

def convertPrefixToPostfix(inp):
    stack = []
    size = len(inp)

    for i in range(size-1,-1,-1):
        curChar = inp[i]

        if isOperator(curChar) == True:
            firstVal = stack.pop()
            secondVal = stack.pop()

            expr = firstVal+secondVal+curChar
            stack.append(expr)

        else:
            stack.append(curChar)

    return stack[-1]

if __name__ == '__main__':
    print(convertPrefixToPostfix('*+AB-CD'))