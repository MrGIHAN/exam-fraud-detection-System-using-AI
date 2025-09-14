import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import Footer from '@/components/shared/Footer';
import UpdateProductForm from '@/components/location/UpdateLocationForm';

const UpdateLocation = () => {
    return (
        <>
            <PageHeader >
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <UpdateProductForm title={"Update Location"} />
                </div>
            </div>
            <Footer />
        </>
    );
}

export default UpdateLocation;