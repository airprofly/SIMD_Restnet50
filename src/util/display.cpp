#include "display.h"
#include "label.h"
#include "param.h"

#include <iostream>
#include <map>

/**
 * @brief display the top 5 labels according to the probability

 * @param cnt the input array of the probability

 */
void display(aligned_unique_ptr<float> &input, int cnt)
{
    std::map<float, int, std::greater<float>> sort_map; // store the probability and the index

    
    for (int i{0}; i < cnt; i++)
    {
        sort_map.emplace(input[i], i); // store the probability and the index
    }
    input.reset(); // free the input array


    std::cout << "the top "<< DISPLAY_CNT <<" labels are: " << std::endl;
    int i{0};
    for (auto it : sort_map)
    {
        std::cout<<" "<<i<<" "<< animalLables.at(it.second) << "  :  " << it.first << std::endl; // print the top 5 labels
        if (++i == DISPLAY_CNT)
            break;
    }

    std::cout << std::endl
              << "the most possible label is " << animalLables.at(sort_map.begin()->second) << std::endl
              << std::endl
              << std::endl; // print the most possible label

    std::cout<<"========================================================================="<<std::endl;

    return;
}