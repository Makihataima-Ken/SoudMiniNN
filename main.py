def main():
    print("Hello from nn-hw!")
    
    def bestClosingTime(customers: str) -> int:
        y = [0]*len(customers)
        n = [0]*len(customers)
        m = len(customers)
        
        wy, wn = 0, 0
        for i in range(m):
            if i == 0:
                if customers[i] == 'Y':
                    y[i] = 1
                    wy = y[i]
                if customers[i] == 'N':
                    n[i] = 1
                    wn = n[i]
            elif customers[i] == 'Y':
                y[i] = wy+1
                wy = y[i]
                n[i] = wn
            elif customers[i] == 'N':
                n[i] = wn+1
                wn = n[i]
                y[i] = wy
            

        ans = 0
        min_penalty = m+1
        for i in range(m):
            current_penalty = (y[m-1]-y[i]) + n[i]
            if customers[i] == 'Y':
                current_penalty +=1
            if current_penalty < min_penalty:
                print(f"Updating min_penalty: {min_penalty} -> {current_penalty} at i: {i}")
                min_penalty = current_penalty
                ans = i

        if ans == 0 and customers[0] == 'N':
            return 0
        
        return ans + 1


    print(bestClosingTime("YYNY"))
    print(bestClosingTime("NNNNN"))
    print(bestClosingTime("YYYY"))
    print(bestClosingTime("YNYY"))
    print(bestClosingTime("NYNNNYYN"))
if __name__ == "__main__":
    main()
