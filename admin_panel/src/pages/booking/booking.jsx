import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import Footer from '@/components/shared/Footer';
import BookingTable from '@/components/booking/BookingTable';

const Booking = () => {
    return (
        <>
            <PageHeader>
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <BookingTable title="Booking List" />
                </div>
            </div>
            <Footer />
        </>
    );
};

export default Booking;
