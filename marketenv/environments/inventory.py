import numpy as np

from ..common.utils import get_sales, get_purchases

class Inventory(object):
    def __init__(self, n_securities, beginning_cash):
        self._n = n_securities
        self._init_cash = beginning_cash
        
        self.reset()
        
    @property
    def positions(self):
        return self._positions
    
    @property
    def cash_balance(self):
        return self._cash_balance
    
    @property
    def cumulative_commissions(self):
        return self._cum_commissions
    
    def reset(self):
        self._positions = np.zeros(self._n, dtype = int)
        self._cash_balance = self._init_cash
        self._cum_commissions = 0
    
    def valid_trade(self, action, current_prices, commission):
        assert (len(action) == self._n), 'TODO' 
        assert (len(current_prices) == self._n), 'TODO'
        
        sale = get_sales(action)
        valid_sale = (sale <= self.positions).all()
        
        purchase = get_purchases(action)
        
        total_purchase_price = (np.dot(purchase, current_prices) 
                                + (commission * sum(purchase > 0)))
        updated_cash_bal = (self.cash_balance + np.dot(sale, current_prices) 
                            - (commission * sum(sale > 0)))
        
        valid_purchase = (total_purchase_price <= updated_cash_bal)
        
        return (valid_sale and valid_purchase)
        
    def update(self, action, current_prices, commission):
        assert (len(action) == self._n), 'TODO' 
        assert (len(current_prices) == self._n), 'TODO'
        
        sale = get_sales(action)
        assert (sale <= self.positions).all(), 'Cannot sell more securities than currently owned'
        
        purchase = get_purchases(action)
        
        sale_commission = commission * sum(sale > 0)
        purchase_commission = commission * sum(purchase > 0)
        
        updated_cash_bal = (self.cash_balance + np.dot(sale, current_prices) 
                            - sale_commission)
        assert (updated_cash_bal > 0.0), 'Not enough cash to cover the sale'
        total_purchase_price = (np.dot(purchase, current_prices) 
                                + purchase_commission)
        
        assert (total_purchase_price <= updated_cash_bal), 'Not enough cash to cover the purchase'
        
        updated_cash_bal -= total_purchase_price
        
        self._cash_balance = updated_cash_bal
        self._positions = self._positions - sale + purchase
        self._cum_commissions += (sale_commission + purchase_commission)