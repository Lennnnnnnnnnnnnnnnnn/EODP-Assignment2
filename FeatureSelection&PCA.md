# Section 2.4, 2.5, 3.4, 3.5, 4.4, 4.5, 5.4, 5.5

**Owner:** Teck (Ben)
**For:** Sabir (report)


## 2.4 Feature Selection

### Filter Method - Mutual Information
This utilises Mutual Information, which analyses the "entropy" (uncertainty) between a target column, price in this scenario as it is being investigated, and the other features in given in preprocessing in order to ascertain the entropy between each pair. This allows inference in the association between each pair, with a higher number indicating more of an association.
This means these features should be investigated further for correlation

The top 3 ended up being: "accommodates: 0.403054", "bedrooms: 0.313188", "room_type_Entire home/apt: 0.292400"
*Note: room_type was converted to numerical data instead of categorical for the library method to function, which results in each of the five room types being assessed as a feature independently*

These scores indicate that pricing is mainly dependent on the number of guests a place holds and the size of the property. Linking back to the question, this suggests ratings do not play an a measurable part.

However, there is a suspicion that this is flawed. "accommodates" and "bedrooms" essentially account for the same thing, and from the data it shows their correlation is at 0.8447, which states they are very correlated
With an assumption that the number of rooms is the primary driver of pricing, both "accomodates" and "bedrooms" are most likely linked to pricing and thus will have high MI scores. This results in redundancy, which means a better feature to measure may be hiding at #4.

### Embedded Method - Decision Trees (Same as Supervised)
Given Trees are better to handle correlated data, this may clear up the result from before.
Start by grouping







Set 1:
price
accommodates 
bathroom
room_type Entire

Set 2:
price
rating_score_value

