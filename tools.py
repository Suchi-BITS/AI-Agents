from langchain.tools import tool
@tool
def add_tool(input: str) -> str:
    """Adds three integers. Input must be a comma-separated string like '1,2,3'."""
    try:
        a, b, c = map(int, input.split(","))
        return str(a + b + c)
    except:
        return "Error: Please provide three comma-separated integers like '1,2,3'"
@tool
def multiply_tool(input: str) -> str:
    """Multiplies three integers. Input must be a comma-separated string like '1,2,3'."""
    try:
        a, b, c = map(int, input.split(","))
        return str(a * b * c)
    except:
        return "Error: Please provide three comma-separated integers like '1,2,3'"
@tool
def findmax_tool(input: str) -> str:
    """Adds three integers. Input must be a comma-separated string like '1,2,3'."""
    try:
        a, b, c = map(int, input.split(","))
        return max(a,b,c)
    except:
        return "Error: Please provide three comma-separated integers like '1,2,3'"

# setup the toolkit
toolkit = [add_tool, multiply_tool, findmax_tool]

