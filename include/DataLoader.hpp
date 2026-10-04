#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

class DataLoader {
public:
    static bool loadCSV(const string& filename, 
                        vector<vector<double>>& X, 
                        vector<vector<double>>& y, 
                        int numTargets) {
                        
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open " << filename << "\n";
            cerr << "Make sure the CSV is in the same folder as your executable.\n";
            return false;
        }

        string line;
        
        while (getline(file, line)) {
            if (line.empty()) continue; 
            
            stringstream ss(line);
            string value;
            vector<double> row;

            while (getline(ss, value, ',')) {
                try {
                    row.push_back(stod(value));
                } catch (...) {
                }
            }

            if (row.size() <= static_cast<size_t>(numTargets)) continue;

            vector<double> x_row(row.begin(), row.end() - numTargets);
            vector<double> y_row(row.end() - numTargets, row.end());

            X.push_back(x_row);
            y.push_back(y_row);
        }
        
        file.close();
        return true;
    }
};
