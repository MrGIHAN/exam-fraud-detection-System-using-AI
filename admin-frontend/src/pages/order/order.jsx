import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import Footer from '@/components/shared/Footer';
import OrderTable from '@/components/order/OrderTable';

const Order = () => {
    return (
        <>
            <PageHeader>
                
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <OrderTable title="Job Application List" />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Order;
