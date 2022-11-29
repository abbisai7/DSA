#iterate the element from back and if operator is encountered, pop last two element from stack, and push operator between them, and push the new expr again into stack.

def isOperator(curChar):
    operators =  ["+","-","*","/","^","%"]
    if curChar in operators:
        return True
    return False

def convertPrefixToInfix(inp):
    stack = []
    size = len(inp)

    for i in range(size-1,-1,-1):
        curChar = inp[i]

        if isOperator(curChar) == True:
            firstVal = stack.pop()
            secondVal = stack.pop()

            expr = '('+firstVal+curChar+secondVal+')'
            stack.append(expr)

        else:
            stack.append(curChar)

    return stack[-1]

if __name__ == '__main__':
    print(convertPrefixToInfix('*+AB-CD'))

