// ai_tycoon.cpp
// A minimal console prototype of "AI Tycoon – The Business Brain"
// C++17, no external deps. Compile: g++ -std=gnu++17 -O2 ai_tycoon.cpp -o ai_tycoon

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <set>
using namespace std;
using namespace std;

// ---------- Utilities ----------
struct Plan {
    double price;
    double adSpend;
    int production; // units to produce (adds to inventory)
};

struct Snapshot {
    int week;
    double baseDemand;     // latent demand signal (unknown to player)
    double eventBoost;     // temporary market boost/shock
    double price;          // chosen price
    double adSpend;        // chosen ad spend
    int production;        // chosen production
    int sold;              // units sold
    int inventoryEnd;      // end-of-week inventory
    double revenue;
    double cost;
    double profit;
};

static std::mt19937_64 rng(12345);

// Clamps for safety
template<class T> T clampv(T v, T lo, T hi) { return max(lo, min(hi, v)); }

// ---------- Market Events ----------
struct MarketEvent {
    string name;
    double baseShock;   // affects baseline demand
    double adShock;     // multiplier on ad effectiveness
    double priceShock;  // multiplier on price sensitivity
};

MarketEvent drawEvent(int week) {
    uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng);
    if (r < 0.10) return {"Viral Trend", +20.0, +0.50, -0.10};
    if (r < 0.20) return {"New Competitor", -15.0, -0.10, +0.25};
    if (r < 0.30) return {"Supply News (positive)", +5.0, +0.05, -0.05};
    if (r < 0.40) return {"Macro Slump", -10.0, -0.10, +0.15};
    return {"Nothing Special", 0.0, 0.0, 0.0};
}

// ---------- Company ----------
struct Company {
    string name = "YouCo";
    int inventory = 40;
    double cash = 20000.0;

    // unit economics
    double unitCost = 8.0;      // production cost per unit
    double fixedCost = 1200.0;  // per turn overhead

    // track history
    vector<Snapshot> history;
};

// ---------- AI Advisor (online linear model) ----------
// Model: demand_hat = w0 + wP*( -price ) + wA*log(1+ad) + wB*baseProxy + wI*inventoryAvail
// where baseProxy is a noisy public proxy the player and AI see (moving avg of sales)
class AIAdvisor {
public:
    AIAdvisor() {
        // initialize weights with small priors
        w0 = 40.0;   // baseline demand guess
        wP = 1.0;    // price sensitivity (higher price -> lower demand, so we apply minus)
        wA = 8.0;    // ad effectiveness on log scale
        wB = 0.5;    // belief in base signal
        wI = 0.1;    // inventory availability small boost
        lr = 0.0015; // learning rate for SGD
    }

    // Suggest plan via simple grid search to maximize predicted profit
    Plan suggest(const Company& c, double baseProxy, double eventAdMult, double eventPriceMult) {
        // sane bounds
        double bestProfit = -1e18;
        Plan best{20, 1000, 50};

        // Grid (coarse for speed; tweak as desired)
        for (double price = 9.0; price <= 40.0; price += 1.0) {
            for (double ad = 0.0; ad <= 8000.0; ad += 500.0) {
                for (int prod = 0; prod <= 120; prod += 10) {
                    double demandHat = predict(price, ad, baseProxy, c.inventory + prod, eventAdMult, eventPriceMult);
                    int canSell = min((int)round(demandHat), c.inventory + prod);
                    double revenue = canSell * price;
                    double cost = prod * c.unitCost + ad + c.fixedCost;
                    double profit = revenue - cost;
                    if (profit > bestProfit) {
                        bestProfit = profit;
                        best = {price, ad, prod};
                    }
                }
            }
        }
        return best;
    }

    // Update weights after observing actual demand (sold units before stockout)
    void learn(double price, double ad, double baseProxy, int inventoryAvail,
               int sold, double eventAdMult, double eventPriceMult)
    {
        // Features
        double x0 = 1.0;
        double xP = - price * (1.0 + eventPriceMult); // higher -> less demand
        double xA = log1p(ad) * (1.0 + eventAdMult);
        double xB = baseProxy;
        double xI = (double)inventoryAvail;

        double yhat = w0*x0 + wP*xP + wA*xA + wB*xB + wI*xI;
        double err = (double)sold - yhat;

        // SGD update
        w0 += lr * err * x0;
        wP += lr * err * xP;
        wA += lr * err * xA;
        wB += lr * err * xB;
        wI += lr * err * xI;

        // keep weights in reasonable ranges to prevent explosions
        w0 = clampv(w0, -200.0, 300.0);
        wP = clampv(wP, -10.0, 10.0);
        wA = clampv(wA, -40.0, 40.0);
        wB = clampv(wB, -5.0, 5.0);
        wI = clampv(wI, -0.5, 0.5);
    }

    // Debugging / transparency
    void printModel() const {
        cout << fixed << setprecision(3);
        cout << "  AI model weights: w0=" << w0
             << ", wP=" << wP << ", wA=" << wA
             << ", wB=" << wB << ", wI=" << wI << "\n";
    }

private:
    double w0, wP, wA, wB, wI; // weights
    double lr;

    double predict(double price, double ad, double baseProxy, int inventoryAvail,
                   double eventAdMult, double eventPriceMult) const
    {
        double x0 = 1.0;
        double xP = - price * (1.0 + eventPriceMult);
        double xA = log1p(ad) * (1.0 + eventAdMult);
        double xB = baseProxy;
        double xI = (double)inventoryAvail;
        double yhat = w0*x0 + wP*xP + wA*xA + wB*xB + wI*xI;
        return max(0.0, yhat);
    }
};

// ---------- Market Simulation ----------
struct Market {
    // Hidden true parameters (player/AI sees only effects)
    double baseDemand = 60.0;        // starts at 60
    double priceSensitivity = 1.4;   // demand drop per $ increase
    double adEffect = 9.0;           // demand lift per log-dollar
    double demandDrift = 0.2;        // weekly drift of baseline (could be +/-)
    double noiseStd = 6.0;

    // Evolve baseline a tad each week
    void drift() {
        normal_distribution<double> n(0.0, 0.8);
        baseDemand = max(5.0, baseDemand + demandDrift + n(rng));
    }

    // Realized demand function
    int realizeDemand(double price, double adSpend, const MarketEvent& ev, int inventoryAvail) {
        // True generative process (unknown to AI)
        double priceMult = 1.0 + ev.priceShock;
        double adMult = 1.0 + ev.adShock;

        double mu = baseDemand + ev.baseShock
                    - priceSensitivity * price * priceMult
                    + adEffect * log1p(adSpend) * adMult
                    + 0.08 * (double)inventoryAvail; // availability slightly boosts conversion

        normal_distribution<double> n(0.0, noiseStd);
        double demand = max(0.0, mu + n(rng));
        return (int)floor(demand + 0.5);
    }
};

// ---------- Game Loop ----------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "==============================\n";
    cout << "  AI TYCOON – The Business Brain\n";
    cout << "==============================\n\n";
    cout << "Goal: Grow profits over 12 turns. Your AI advisor learns and suggests a plan each week.\n";
    cout << "You sell a single product. Unit production cost = $8. Fixed weekly overhead = $1200.\n";
    cout << "You begin with 40 units in inventory and $20,000 cash.\n\n";

    Company co;
    Market mk;
    AIAdvisor ai;

    int weeks = 12;
    double baseProxy = 50.0; // what player/AI "believes" baseline demand might be (public noisy proxy)

    for (int week = 1; week <= weeks; ++week) {
        cout << "\n==== Week " << week << " ====\n";
        mk.drift();
        MarketEvent ev = drawEvent(week);
        cout << "Market event: " << ev.name << "\n";

        // AI suggestion
        Plan plan = ai.suggest(co, baseProxy, ev.adShock, ev.priceShock);
        cout << fixed << setprecision(2);
        cout << "AI suggests -> Price: $" << plan.price
             << " | Ad: $" << plan.adSpend
             << " | Produce: " << plan.production << " units\n";
        ai.printModel();

        // Player choice
        cout << "Accept AI plan? (y/n) ";
        string yn; getline(cin, yn);
        if (yn.size() == 0) { yn = "y"; }
        Plan chosen = plan;
        if (yn[0] == 'n' || yn[0] == 'N') {
            cout << "Enter your Price [$9..$40]: ";
            string s; getline(cin, s);
            if (!s.empty()) chosen.price = clampv(stod(s), 9.0, 40.0);

            cout << "Enter your Ad Spend [$0..$10000]: ";
            getline(cin, s);
            if (!s.empty()) chosen.adSpend = clampv(stod(s), 0.0, 10000.0);

            cout << "Enter your Production [0..200]: ";
            getline(cin, s);
            if (!s.empty()) chosen.production = (int)clampv(stoi(s), 0, 200);
        }

        // Apply production (pay costs immediately)
        co.inventory += chosen.production;

        // Realize sales
        int potential = mk.realizeDemand(chosen.price, chosen.adSpend, ev, co.inventory);
        int sold = min(potential, co.inventory);
        co.inventory -= sold;

        // Finance
        double revenue = sold * chosen.price;
        double cost = chosen.production * co.unitCost + chosen.adSpend + co.fixedCost;
        double profit = revenue - cost;
        co.cash += profit;

        // Record snapshot
        Snapshot snap{
            week,
            mk.baseDemand,
            ev.baseShock,
            chosen.price,
            chosen.adSpend,
            chosen.production,
            sold,
            co.inventory,
            revenue,
            cost,
            profit
        };
        co.history.push_back(snap);

        // Update AI on the observed outcome
        ai.learn(chosen.price, chosen.adSpend, baseProxy, co.inventory + sold, sold, ev.adShock, ev.priceShock);

        // Update public baseline proxy (what players can infer)
        // Use moving average of last 3 weeks' sales as a noisy "market temperature"
        int start = max(0, (int)co.history.size() - 3);
        double avgSales = 0.0;
        for (int i = start; i < (int)co.history.size(); ++i) avgSales += co.history[i].sold;
        avgSales /= (int)co.history.size() - start;
        // Blend with a little random noise to simulate imperfect info
        normal_distribution<double> n(0.0, 3.0);
        baseProxy = max(0.0, 0.70 * baseProxy + 0.30 * avgSales + n(rng));

        // HUD
        cout << fixed << setprecision(2);
        cout << "\n— Results —\n";
        cout << "Sold: " << sold << " units | Revenue: $" << revenue << "\n";
        cout << "Costs: $" << cost << " | Profit: $" << profit << "\n";
        cout << "End Inventory: " << co.inventory << " | Cash: $" << co.cash << "\n";
        cout << "Market baseline (hidden true): " << mk.baseDemand
             << " | Your inferred proxy: " << baseProxy << "\n";

        if (co.cash < -5000.0) {
            cout << "\nYou ran out of cash. Game over early.\n";
            break;
        }
    }

    // Post-game summary
    cout << "\n================ SUMMARY ================\n";
    double totalProfit = 0.0;
    int totalSales = 0;
    for (auto &s : co.history) {
        totalProfit += s.profit;
        totalSales += s.sold;
    }
    cout << fixed << setprecision(2);
    cout << "Total Profit: $" << totalProfit << " | Total Units Sold: " << totalSales << "\n";
    cout << "Final Cash: $" << co.cash << " | Final Inventory: " << co.inventory << "\n";
    cout << "Thanks for playing AI Tycoon!\n";
    return 0;
}
